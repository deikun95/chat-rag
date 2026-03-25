# Chat with Docs — Interview Preparation

30 questions likely to be asked when an interviewer sees this RAG pet project on an AI Engineer resume, with detailed answers specific to this codebase.

---

## Architecture & Design Decisions

### 1. Why did you choose this specific chunking strategy (recursive split)? What are the tradeoffs vs. fixed-size or semantic chunking?

`_recursive_split` in `ingestion_service.py` uses a separator hierarchy: `["\n\n", "\n", ". ", " "]`. It tries paragraph breaks first, then lines, then sentences, then words, with a hard character fallback as last resort. This preserves semantic coherence — a paragraph boundary is more meaningful than an arbitrary cutoff. **vs fixed-size**: fixed-size is simpler but cuts mid-sentence, destroying meaning at boundaries. **vs semantic chunking** (using an embedding model to detect topic shifts): semantic is more accurate but requires model inference at chunk time, making ingestion much slower and expensive. Recursive split is the pragmatic sweet spot for this scale. **Improvement**: use spaCy for sentence splitting instead of naive `.`  which breaks on "Dr. Smith".

### 2. Why ChromaDB over pgvector/Pinecone/Weaviate? When would you switch?

ChromaDB is zero-infrastructure — `pip install`, point to a directory, done. No server, no cloud account. Perfect for a portfolio project. **I'd switch to**:

- **pgvector** if I already had PostgreSQL (one DB for metadata + vectors, transactional consistency)
- **Pinecone** for managed serverless (zero-ops at scale)
- **Qdrant/Weaviate** for production with multi-tenancy, replication, filtered search

ChromaDB breaks with concurrent writes (SQLite underneath is single-writer) and has no replication/backup.

### 3. Why cosine similarity? When would you use dot product or Euclidean distance?

Configured in `vector_store.py` line 28: `{"hnsw:space": "cosine"}`. OpenAI's `text-embedding-3-small` outputs normalized vectors, so cosine measures angle between vectors regardless of magnitude — robust for comparing texts of different lengths.

- **Dot product**: equivalent when vectors are normalized (slightly faster, no normalization step)
- **Euclidean**: better when absolute position matters (image embeddings, geographic data)

For text retrieval with normalized embeddings, cosine and dot product are effectively interchangeable.

### 4. How would you handle a 500-page PDF? What breaks at scale?

Current problems: `extract_pages` iterates all pages sequentially, `embed_texts` batches at `MAX_BATCH_SIZE=100`, so a 500-page PDF (~2500 chunks) means ~25 API calls. The upload endpoint is synchronous — blocks the HTTP request while processing. What breaks:

1. Client timeout (processing could take 60+ seconds)
2. OpenAI rate limits on embedding calls
3. `max_upload_size_mb=20` rejects many large PDFs
4. Full file loaded into memory via `await file.read()`

**Fix**: background task queue (Celery/ARQ), return 202 immediately with a status polling endpoint, chunked file upload, async embedding calls.

### 5. Why embed the source metadata ("[Source: doc, Page N]") directly into the chunk text?

In `ingestion_service.py` lines 136-139, `[Source: {doc_name}, Page {N}]` is prepended to chunk text before embedding. This means the LLM can see where info came from without a separate lookup.

**Downside**: it pollutes the embedding vector — part of the vector now represents "Page 5" rather than the actual content, slightly degrading retrieval quality. The metadata is already stored separately in the `metadatas` dict (lines 186-194), and `_build_context` in `chat_service.py` already adds `[Source {i}]` labels. So this is **double-encoding** — I'd remove it from the embedded text and only add it at LLM prompt time.

---

## RAG Pipeline Deep Dive

### 6. Your relevance threshold is 0.05 — how did you pick that number? How would you tune it properly?

`min_score = 0.05` in `retrieval_service.py` is extremely permissive — only filters vectors nearly orthogonal to the query. It was set pragmatically to avoid filtering useful results. **Proper tuning**:

1. Build an evaluation dataset (question -> correct source chunk)
2. Log retrieval scores for real queries
3. Plot score distributions for relevant vs. irrelevant chunks
4. Pick the threshold that maximizes F1 (or recall at acceptable precision)
5. Consider **dynamic thresholds** — take results within 80% of top score

A well-tuned cosine threshold with OpenAI embeddings is typically 0.3-0.5.

### 7. What happens when a user asks a question that spans information across multiple chunks from different pages?

The system handles this well. `retrieve()` fetches `top_k=5` from any page/document. `_build_context` numbers them `[Source 1]`, `[Source 2]`, and the LLM can synthesize across them and cite each. **Weakness**: chunks are retrieved independently by similarity to the query. If the answer requires "revenue" chunk + "expenses" chunk but the query only says "profit," one might be missed. **Fix**: query decomposition (break complex questions into sub-queries) or HyDE (generate a hypothetical answer, embed that, retrieve).

### 8. How would you add a reranker? Where in the pipeline does it go and why?

It goes in `retrieval_service.py` between vector search (line 46) and final filtering (line 68). After getting `candidates` from `vector_store.search`, pass each query+candidate pair through a cross-encoder (e.g., Cohere Rerank or `bge-reranker`), re-sort by reranker scores, then apply `min_score` and `top_k`. This is the classic **two-stage retrieval** pattern — fast approximate retrieval via bi-encoder, then slow accurate reranking via cross-encoder on top ~20 candidates. The code already fetches `top_k * 2` candidates (`search_k` on line 45) — pre-architected for reranking. Typically improves precision 5-15%.

### 9. What's the difference between your embedding model (text-embedding-3-small) and a cross-encoder? When do you use each?

`text-embedding-3-small` is a **bi-encoder**: maps each text independently into a 1536-dim vector. Fast — embed documents once at ingestion, only embed query at search time. A **cross-encoder** takes query+document as a single input pair and outputs a relevance score directly, attending to interactions between texts. Much more accurate (can understand negation, implicit relationships) but O(n) per query — must run inference for every candidate. In this codebase, the bi-encoder does initial retrieval; a cross-encoder would be added as a reranker over top-k results. You never use a cross-encoder for full corpus search because it doesn't produce storable embeddings.

### 10. Your chunk overlap is 200 chars — what happens if you set it to 0? To 1000?

`CHUNK_OVERLAP=200` means each new chunk starts with the last 200 chars of the previous one.

- **Overlap=0**: sentences at chunk boundaries get cut. A sentence spanning two chunks is in neither completely — unretrievable for queries about that specific info.
- **Overlap=1000** (half of `CHUNK_SIZE=2000`): massive redundancy. Each text appears in 2-3 chunks, doubling storage/embedding cost. Retrieval returns near-duplicate chunks wasting the LLM context window.
- **Current 200 chars (~10%)**: reasonable balance. I'd consider 300-400 and add deduplication in retrieval to merge overlapping results.

---

## Embeddings & Vector Search

### 11. What is an embedding? Explain what text-embedding-3-small actually outputs.

A dense numerical representation of text in high-dimensional space where semantically similar texts are close together. When `embed_texts` calls `client.embeddings.create(model="text-embedding-3-small", input=batch)`, OpenAI returns a list of 1536-dimensional float vectors. Each dimension captures a learned semantic feature — not individually interpretable, but collectively encoding meaning such that "dog" and "puppy" are close while "dog" and "algebra" are far. Critically, the **same model** embeds both document chunks (during ingestion) and user queries (at search time) — they must share the same embedding space.

### 12. Why 1536 dimensions? What happens if you reduce to 256?

1536 is `text-embedding-3-small`'s default. OpenAI supports a `dimensions` parameter to truncate using Matryoshka Representation Learning. Reducing to 256 would:

- **6x less storage** in ChromaDB
- **Faster similarity search** (HNSW distance is O(d))
- **Information loss** — fine-grained semantic distinctions live in higher dimensions

For many use cases, 256-512 retains 90-95% quality. Currently the `dimensions` parameter isn't passed in `embed_texts` — I'd make it configurable and benchmark retrieval quality at 512 and 256.

### 13. How does HNSW (the index ChromaDB uses) work? What are the tradeoffs vs. brute-force search?

Hierarchical Navigable Small World — a multi-layer graph where nodes are vectors and edges connect nearby ones. Top layers are sparse (long-range "highway" connections), bottom layers are dense (local precision). Search starts at the top, greedily navigates toward the query, descends to finer layers. **O(log n) search vs O(n) brute-force**, with 90-99% recall. **Tradeoffs**: more memory (graph structure), build-time cost on insert, approximate (can miss true nearest neighbor). For this project's scale (thousands of chunks) brute-force would actually work fine — HNSW benefits kick in at 100k+ vectors.

### 14. If two documents contradict each other, how does your system handle it?

System prompt rule 5: "If sources contradict each other, mention the contradiction." Both contradicting chunks may be retrieved since `retrieve()` selects by similarity without consistency checking. `_build_context` presents all as numbered sources, so the LLM sees both and can compare. **Limitation**: relies entirely on LLM detecting contradictions — no programmatic detection. If only one contradicting chunk gets retrieved, the conflict goes unnoticed. **Improvement**: add document timestamp metadata to prefer newer info, or implement explicit contradiction detection comparing retrieved chunks before sending to LLM.

---

## LLM & Prompting

### 15. Walk me through your system prompt. Why each rule?

- **Rule 1** (only use context): prevents hallucination, grounds the LLM
- **Rule 2** (explain what docs DO cover): better UX than "I don't know" — user learns what to ask
- **Rule 3** (cite [Source N]): enables verifiability via source references in the SSE `sources` event
- **Rule 4** (be concise): prevents verbose responses burying the answer
- **Rule 5** (mention contradictions): handles multi-document conflicts
- **Rule 6** (preserve accuracy): prevents rounding "$4,237,891" to "about $4 million"
- **Rule 7** (match language): multilingual support — question in Spanish -> answer in Spanish

**I'd add**: rules about handling tabular data and not inventing source numbers that don't exist.

### 16. How does your conversation memory work? What's the token limit risk?

History is built client-side in `useChat.ts` from all completed messages, sent as `history` in `ChatRequest`, truncated server-side to `MAX_HISTORY_MESSAGES=20`. Each message is added to the LLM messages array. **Token risk**: 20 messages, each potentially containing full RAG context (5 chunks x ~500 tokens), plus system prompt — easily 50k+ tokens. `gpt-4o-mini` has 128k context so unlikely to overflow, but **costs scale linearly** with input tokens. No server-side session storage — refresh loses history.

### 17. You send the full history every request — how would you optimize for long conversations?

Current: crude truncation to last 20 messages. Better approaches:

1. **Summarize** — every N turns, LLM summarizes conversation into a compact message replacing older ones
2. **Strip RAG context from history** — don't send the full "[Source:...]" content back, just the clean answer
3. **Server-side sessions** — backend stores history with conversation ID, client only sends new question
4. **Sliding window with importance** — keep explicitly referenced messages

The biggest quick win: strip RAG context from historical assistant messages before re-sending.

### 18. How would you evaluate answer quality? What metrics would you use?

**Retrieval metrics** (deterministic):

- **Recall@K**: does the correct source chunk appear in top-k?
- **MRR**: how high is the correct chunk ranked?

**Answer metrics** (LLM-judged):

- **Faithfulness**: every claim traceable to a source chunk?
- **Relevance**: does it address the question?
- **Citation accuracy**: do [Source N] references match actual info?

**Tools**: RAGAS framework for automated faithfulness/relevance scoring, LLM-as-judge (GPT-4 grading against ground truth). **User metrics**: thumbs up/down, follow-up rate, source click-through.

### 19. How would you prevent hallucinations beyond what your prompt does?

1. **Verification step** — second LLM call checking if each claim is supported by context
2. **Citation verification** — parse [Source N] and verify cited source actually contains claimed info
3. **Lower temperature** (currently using default ~1.0, not set in code) — reduces creative generation
4. **Early exit** — if retrieval returns nothing after filtering, return a structured "no info" response instead of letting the LLM improvise from "No relevant context found"
5. **RAG confidence score** based on retrieval scores, shown to user
6. **Constrained decoding** / logit bias to reduce out-of-context generation

---

## Security

### 20. How is your system vulnerable to prompt injection through uploaded PDFs?

Significantly vulnerable. Raw PDF text goes through `page.get_text("text")` with only whitespace cleanup, no adversarial content sanitization. This text is embedded, stored, retrieved, and injected directly into the LLM prompt via `_build_user_message`. A malicious PDF with "Ignore all previous instructions..." would be treated as regular content. **Mitigations**:

1. Sanitize extracted text for instruction-like patterns
2. Use XML delimiters (`<context>...</context>`) with explicit instructions to treat content as data
3. Content classification at upload time
4. Output guardrails detecting compromised behavior

### 21. What if someone uploads a PDF with instructions like "ignore previous instructions"?

It passes all validation (which only checks file integrity, not content), gets chunked, embedded, stored. When retrieved, the malicious instruction appears inside the user message. **Zero mitigations currently.** Defenses:

1. Content scanning at upload with a prompt injection classifier
2. XML-style delimiters separating data from instructions
3. OpenAI moderation API on extracted text
4. Output guardrails
5. System messages generally take priority over user content in modern models, providing some inherent resilience

### 22. How would you add authentication/rate limiting?

`rate_limit_rpm=20` exists in config but is never enforced. **Rate limiting**: `slowapi` middleware keyed by IP/API key. **Auth**: JWT-based with `/api/auth/login`, `get_current_user` dependency on protected routes. Frontend `api.ts` would include token in headers. For a quick MVP, API key header auth (check against stored hash). Add per-user rate limits and upload quotas in the database.

---

## Production & Scale

### 23. How would you handle concurrent uploads? What breaks with 100 simultaneous users?

Problems:

- SQLite single-writer lock -> "database is locked" errors
- ChromaDB also uses SQLite -> double locking
- Module-level singleton services -> shared across requests (OK for OpenAI client, bad for DB)
- Embedding calls hit OpenAI rate limits (no global coordination)
- `upload_document` blocks until full pipeline completes (30+ seconds for large PDFs)

**Fix**: background job queue (Celery + Redis), return 202 Accepted, replace SQLite with PostgreSQL, use client-server vector DB.

### 24. Your SQLite + ChromaDB setup — what would you change for production?

- **SQLite -> PostgreSQL** with SQLAlchemy + Alembic migrations (connection pooling, concurrent writes, ACID)
- **ChromaDB -> pgvector** (simplest — one DB for everything) or **Qdrant/Weaviate** (dedicated vector DB with replication) or **Pinecone** (managed, zero-ops)
- Add **Redis** for query caching and job broker
- Proper secrets management (AWS Secrets Manager) instead of `.env` file

### 25. How would you add observability? What would you log/monitor?

**Logging**: structured JSON with correlation IDs, log every retrieval with scores, every LLM call with token counts/latency.

**Metrics** (Prometheus): request latency p50/p95/p99, embedding API latency, retrieval score distributions, cost per query, ChromaDB collection size.

**Tracing** (OpenTelemetry): full pipeline spans — upload->extract->chunk->embed->store and query->embed->retrieve->LLM->stream.

**Key alerts**: average top-1 score trending down, embedding error rates, queries where filtering removed all results, cost anomalies.

### 26. How would you deploy this? Walk me through the infrastructure.

1. **Frontend**: Vercel (or static export behind CDN)
2. **Backend**: Dockerized FastAPI, 2+ replicas behind ALB/Cloudflare
3. **Database**: RDS PostgreSQL with pgvector
4. **Queue**: Redis + Celery workers for ingestion
5. **Storage**: S3 for original PDFs
6. **CI/CD**: GitHub Actions -> build -> test -> deploy to staging/prod

SSE streaming requires load balancer to support long-lived connections. The `X-Accel-Buffering: no` header handles Nginx.

### 27. What's the cost per query? How would you optimize it?

- **Embedding query**: ~$0.000002 (negligible)
- **LLM**: system prompt (~~150 tokens) + context (5 chunks x ~500 = ~2500) + history + question -> **~~$0.001-0.005** per query with gpt-4o-mini
- **Vector search**: free (local)

**Optimize**: cache frequent queries, reduce top_k from 5->3 if quality permits, summarize history, use smaller dimensions for embeddings. Biggest cost driver: history in long conversations (20 messages = 10k+ tokens).

---

## Code Quality

### 28. Why separate services (DocumentService, EmbeddingService, etc.) instead of one monolith?

Single Responsibility: `DocumentService` doesn't know about embeddings, `EmbeddingService` doesn't know about chunks, `IngestionService` orchestrates. Enables: swapping ChromaDB->Pinecone (change only `vector_store.py`), unit testing `_recursive_split` without an API key, independent scaling.

**Problem**: `documents.py` and `chat.py` both create their own `EmbeddingService()` and `VectorStoreService()` singletons — duplicate instances. **Fix**: FastAPI's `Depends()` with a proper DI container.

### 29. How would you test the RAG pipeline? What does a good test look like when LLM outputs are non-deterministic?

Three levels:

1. **Unit** (deterministic): test `_recursive_split` with known inputs, `validate_file` with bad PDFs, `_build_context` format
2. **Integration** (semi-deterministic): upload known PDF -> verify chunk count -> verify retrieval returns right chunks
3. **E2E/LLM** (non-deterministic): assertion-based — upload PDF with "revenue was $5.2M", ask about revenue, assert answer **contains** "$5.2M" and a `[Source` citation and **doesn't contain** info not in the document. Use `temperature=0` for reproducibility. Golden set + LLM-as-judge for regression testing.

### 30. What would you change if you rebuilt this from scratch?

1. **Async everywhere** — `AsyncOpenAI` instead of sync client blocking the event loop
2. **Background processing** — task queue for ingestion, not in HTTP request
3. **Dependency injection** — FastAPI `Depends()` instead of duplicate module-level singletons
4. **Remove embedded metadata from chunk text** — keep only in metadata dict to avoid polluting embeddings
5. **Hybrid search** — vector + BM25 for exact terms (product names, IDs)
6. **Add reranker** in retrieval pipeline
7. **PostgreSQL + pgvector** from day one (eliminate dual-database)
8. **Backend streaming cancellation** — currently `stop` in frontend aborts fetch but backend keeps generating
9. **Auth + document ownership** from the start

---

## Appendix A: Is This Project Enough for an AI Engineer Interview?

This project + deep understanding of the 30 questions above covers **70-80%** of what you need for an AI Engineer interview. It's a strong foundation.

### What this project covers well

- RAG architecture end-to-end
- Embeddings, vector search, chunking strategies
- LLM API integration, prompt engineering
- Streaming, SSE, full-stack integration
- System design thinking (the "what would you change" questions)

This is enough for **junior/mid AI Engineer roles** where the job is mostly building LLM-powered applications.

### What's missing (but worth knowing at surface level)

**1. ML/AI fundamentals (surface level only):**

- How do transformers work? (attention mechanism intuition, not the math)
- What is fine-tuning vs RAG vs prompt engineering? When to use which?
- What are tokens? How does tokenization work? Why does it matter?
- Explain temperature, top-p, top-k sampling
- What is context window and why does it matter architecturally?

**2. Evaluation & experimentation:**

- How do you A/B test an AI feature?
- How do you measure if your RAG is actually good? (not just "I'd use RAGAS" — actually doing it)
- What is an eval suite and how do you build one?

**3. Agents & tool use (hot topic in 2025-2026):**

- Function calling / tool use with LLMs
- Agent loops (ReAct, plan-and-execute)
- When to use agents vs simple RAG vs fine-tuning

**4. Production AI patterns:**

- Guardrails and content moderation
- Caching strategies for LLM apps
- Cost optimization at scale
- Monitoring LLM quality drift over time
- Handling model version upgrades without breaking things

**5. Data pipeline experience:**

- Working with messy real-world documents (OCR, tables, images in PDFs)
- Data cleaning and preprocessing at scale
- ETL for AI systems

### The one thing to still add

Run an actual eval on the project. Upload 3-5 real documents, write 15-20 questions with expected answers, measure how many the bot gets right. If an interviewer asks "how good is your RAG?" and you say "78% accuracy on 20 test questions, main failures were multi-hop questions across pages" — that separates you from everyone else who just built a demo and never measured it.

---

## Appendix B: AI Engineer vs ML Engineer — What You Actually Need to Know

### The distinction is real

**ML Engineer**: trains models, fine-tunes, understands gradients, loss functions, GPU infra, datasets, evaluation at model level.

**AI Engineer**: builds products WITH models — API integration, RAG, agents, prompt engineering, full-stack, shipping features to users.

You don't need to know backpropagation math or how to train a LoRA adapter.

### But you need intuition, not math

When something breaks — and it will — you need to debug it:

- Your RAG returns garbage results. Is it a chunking problem? Embedding model problem? Retrieval threshold? Prompt problem? You need enough understanding to diagnose where in the pipeline the issue is. **This project teaches this.**
- Your boss says "should we fine-tune or use RAG?" You need to know the tradeoffs to give a real answer.
- A new model comes out (GPT-5, Claude 5, Gemini 3). You need to evaluate: is it worth switching? What changes? Does your architecture hold?

### What you can skip entirely

- Training from scratch
- Dataset curation for training
- GPU/TPU infrastructure
- Model architecture research
- Math behind attention (just know the intuition)
- Fine-tuning mechanics (know WHEN to use it, not HOW to do it)

### What interviewers actually ask (% breakdown)


| Topic                                  | Weight | Covered by this project? |
| -------------------------------------- | ------ | ------------------------ |
| System design & architecture           | 40%    | Yes                      |
| RAG / embeddings / vector search depth | 25%    | Yes                      |
| Product thinking & tradeoffs           | 20%    | Yes (Q&A above)          |
| Agents / tool use concepts             | 10%    | Know the concepts        |
| ML fundamentals (surface level)        | 5%     | Minimal study needed     |


### Bottom line

This project is your **anchor story** — the thing you walk through in detail, showing depth. Know it deeply, know the tradeoffs, have real eval numbers, and you're ready.