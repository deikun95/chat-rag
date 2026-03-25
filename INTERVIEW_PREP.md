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

---

## Appendix C: General AI Engineer Interview Questions (Beyond This Project)

These are common questions asked at AI Engineer interviews that go beyond RAG. Grouped by topic.

---

### LLM Fundamentals

#### 31. What is a transformer? Explain the attention mechanism in simple terms.

A transformer processes all tokens in parallel (unlike RNNs which go one-by-one). The core idea is **attention**: for each word, the model asks "which other words in this text should I pay attention to?" It computes a relevance score between every pair of tokens, then creates a weighted mix of their representations. "The cat sat on the mat because **it** was tired" — attention lets the model figure out "it" refers to "cat" by giving "cat" a high attention weight. **Multi-head attention** runs this process multiple times in parallel, each head learning different relationships (one head for grammar, another for meaning, etc.). You don't need to know the Q/K/V matrix math — just this intuition.

#### 32. What are tokens? Why does tokenization matter?

Tokens are the units LLMs actually process — not words, not characters, but subword pieces. "unhappiness" might become ["un", "happiness"]. GPT uses BPE (Byte Pair Encoding) which builds a vocabulary of common subword units. **Why it matters for AI Engineers**: (1) pricing — you pay per token, so knowing that "hello" is 1 token but " hello" (with space) is 2 tokens affects cost estimates, (2) context window — a 128k token limit is roughly 96k words, not 128k words, (3) multilingual — non-English text often uses 2-4x more tokens per word, which means shorter context and higher cost, (4) structured data like JSON is very token-heavy.

#### 33. Explain temperature, top-p, and top-k sampling.

These control randomness in LLM outputs. **Temperature**: scales the probability distribution. Temperature=0 always picks the highest-probability token (deterministic). Temperature=1 is the default distribution. Temperature=2 flattens it (more random/creative). **Top-k**: only consider the top k most likely tokens, discard the rest. Top-k=1 is greedy (same as temp=0). Top-k=50 considers 50 candidates. **Top-p (nucleus sampling)**: instead of a fixed count, keep tokens until their cumulative probability reaches p. Top-p=0.9 means "keep the smallest set of tokens that covers 90% probability." For RAG/factual apps, use low temperature (0-0.3). For creative writing, higher (0.7-1.0).

#### 34. What is the context window? Why does it matter architecturally?

The context window is the maximum number of tokens the model can process in a single call (input + output combined). GPT-4o-mini has 128k tokens. **Why it matters**: (1) your entire prompt — system message + retrieved chunks + conversation history + user question — must fit, (2) longer context = higher cost (linear with token count), (3) models perform worse on information in the middle of very long contexts ("lost in the middle" problem), (4) it constrains how much history you can send in a chat, how many chunks you can retrieve, how large your documents can be. Architecturally, this is why you need chunking, summarization, and retrieval rather than just dumping entire documents into the prompt.

#### 35. What is fine-tuning vs RAG vs prompt engineering? When to use which?

**Prompt engineering**: write better instructions/examples in the prompt. Free, instant, no training. Use first for everything. Limitations: can't teach the model new knowledge or change its behavior fundamentally.

**RAG**: retrieve relevant documents and inject them into the prompt. Gives the model access to your private/current data without training. Use when you need answers grounded in specific documents, when data changes frequently, or when you need citation/attribution. Your project is this.

**Fine-tuning**: train the model on your data to change its behavior, style, or teach it domain-specific patterns. Use when you need a specific output format consistently, domain-specific terminology, or when prompt engineering hits its limits. Expensive, slow to iterate, requires training data.

**Decision tree**: Try prompt engineering first -> if you need external knowledge, add RAG -> if the model still doesn't behave right (wrong tone, format, domain understanding), fine-tune -> if fine-tuning isn't enough, consider training a custom model (rare for AI Engineers).

---

### Agents & Tool Use

#### 36. What is function calling / tool use? How does it work?

Function calling lets an LLM decide to invoke external tools (APIs, databases, calculators) instead of generating text. You define tools with names, descriptions, and parameter schemas. The LLM outputs a structured JSON saying "call function X with arguments Y" instead of a text response. Your code executes the function, returns the result, and the LLM incorporates it into its answer. Example: user asks "What's the weather in London?" — the LLM outputs `{"function": "get_weather", "args": {"city": "London"}}`, your code calls a weather API, returns "15C rainy", and the LLM says "It's 15C and rainy in London." This is how ChatGPT plugins, code interpreter, and most AI assistants work.

#### 37. What is an AI agent? How is it different from a simple LLM call?

A simple LLM call: input -> output, done. An **agent** runs in a loop: observe -> think -> act -> observe -> think -> act, repeating until the task is done. The LLM decides which actions to take, evaluates results, and adjusts its plan. **ReAct pattern**: the agent alternates between reasoning ("I need to find the user's order") and acting (calling an API). **Plan-and-execute**: the agent first creates a full plan, then executes steps one by one. Agents can handle multi-step tasks ("research this topic, compare 3 options, write a report") that a single LLM call can't. The risk: agents can loop forever, take wrong actions, or accumulate errors. You need guardrails, timeouts, and human-in-the-loop for production.

#### 38. When would you use an agent vs simple RAG vs a pipeline?

**Simple RAG** (like this project): user asks a question, you retrieve context, LLM answers. Best for Q&A over documents, support chatbots, knowledge bases. Predictable, fast, cheap.

**Pipeline** (fixed chain of LLM calls): input -> summarize -> extract entities -> format output. Each step is predefined. Best when the workflow is known and doesn't change. Example: "process every email: classify intent, extract action items, create tasks."

**Agent** (dynamic tool use in a loop): the LLM decides what to do next based on results. Best when the task requires dynamic decision-making, multiple tools, or the steps aren't known in advance. Example: "research this company and prepare a competitive analysis" — the agent decides which sources to check, what to compare, when it has enough info.

**Rule of thumb**: use the simplest approach that works. Most production AI features are pipelines or RAG, not agents.

---

### Evaluation & Quality

#### 39. How do you know if your AI feature is working well? What metrics do you track?

**Offline metrics** (before deployment): accuracy on test set, faithfulness score (RAGAS), retrieval recall@k, latency p95. Run these in CI on every change.

**Online metrics** (in production): (1) **task completion rate** — did the user get what they needed? (2) **thumbs up/down** — explicit feedback, (3) **retry rate** — user rephrasing the same question = bad answer, (4) **fallback rate** — how often the bot says "I don't know", (5) **latency** — time to first token, total response time, (6) **cost per interaction**.

**Leading indicators of quality degradation**: average retrieval score trending down (documents getting stale?), increased retry rate, higher "I don't know" rate, token usage per query increasing (history bloat?).

#### 40. What is an eval suite and how do you build one?

An eval suite is a set of test cases (input -> expected output) that you run automatically to measure AI quality. **How to build**: (1) start with 20-50 real user questions from logs or product scenarios, (2) for each question, write the expected answer and which source document it should come from, (3) define scoring: exact match for facts/numbers, LLM-as-judge for open-ended answers, retrieval recall for source verification, (4) automate it — run on every PR, track scores over time, alert on regression. **Key insight**: LLM outputs are non-deterministic, so use `temperature=0` and assertion-based checks ("answer contains X") rather than exact string matching. Tools: RAGAS, DeepEval, or just pytest + OpenAI calls for judging.

#### 41. What is LLM-as-judge? When is it reliable?

Using one LLM to evaluate another LLM's output. You give GPT-4 the question, the expected answer, and the actual answer, and ask it to score on criteria like correctness, relevance, completeness. **When reliable**: comparing two answers (A vs B), evaluating factual consistency against a reference, scoring on well-defined rubrics. **When unreliable**: subjective quality ("is this creative?"), self-evaluation (same model judging itself), edge cases the judge model doesn't understand. **Best practices**: use a stronger model as judge (GPT-4 judging GPT-4o-mini), provide clear scoring rubrics, always validate against human judgments on a sample, use multiple judges and average.

---

### Production Patterns

#### 42. How do you handle LLM API failures in production?

(1) **Retries with exponential backoff** — the OpenAI SDK does this automatically (3 retries). (2) **Fallback models** — if GPT-4o-mini is down, fall back to GPT-4o or Claude. (3) **Circuit breaker** — after N consecutive failures, stop calling the API for X seconds to avoid hammering a broken service. (4) **Graceful degradation** — show "AI is temporarily unavailable" instead of crashing. (5) **Timeouts** — set reasonable timeouts (30s for streaming) so users don't wait forever. (6) **Queue + async** — for non-real-time features, queue requests and process when the API is available. In this project, `chat_service.py` catches exceptions and returns an SSE error event, which is a good start but doesn't implement retries or fallbacks.

#### 43. How do you cache LLM responses? What are the tradeoffs?

**Exact match cache**: hash the full prompt, cache the response. Works for identical queries. Simple but low hit rate — even a small change in context (different retrieved chunks) means a cache miss. **Semantic cache**: embed the query, find similar cached queries by vector similarity. Higher hit rate but risk of returning stale/wrong answers for similar-but-different questions. **Tradeoffs**: (1) stale data — cached answers don't reflect document updates, (2) personalization — cached answer for user A might be wrong for user B if they have different documents, (3) streaming — you need to cache the full response and re-stream it. **Best approach for RAG**: cache at the retrieval level (same query -> same chunks) rather than the LLM level. Chunks change less often and the cache is more reusable.

#### 44. How do you handle model version upgrades without breaking things?

(1) **Pin model versions** — use `gpt-4o-mini-2024-07-18` not `gpt-4o-mini` so updates don't surprise you. (2) **Run eval suite** against the new model version before switching. (3) **Shadow mode** — send traffic to both old and new model, compare outputs, don't show new model to users yet. (4) **Gradual rollout** — 5% of traffic to new model, monitor metrics, increase if quality holds. (5) **Prompt may need adjustment** — a new model might interpret instructions differently, so test your prompts. In this project, the model is hardcoded as `"gpt-4o-mini"` in `chat_service.py` — I'd move it to config and add a model version field.

#### 45. What are guardrails and how do you implement them?

Guardrails are checks that run before and after LLM calls to ensure safety and quality. **Input guardrails** (before LLM): PII detection (block/redact sensitive data), topic classification (reject off-topic queries), prompt injection detection, rate limiting. **Output guardrails** (after LLM): toxicity/harmful content detection (OpenAI moderation API), hallucination check (verify claims against sources), format validation (is the output valid JSON if expected?), PII leakage check (model shouldn't expose data from other users). **Implementation**: middleware pattern — wrap the LLM call with pre/post processors. Libraries: Guardrails AI, NeMo Guardrails, or custom with classification models. In production, guardrails add 100-500ms latency, so balance safety vs speed.

---

### System Design

#### 46. Design a customer support chatbot that uses your company's knowledge base. Walk me through the architecture.

**Components**: (1) **Ingestion pipeline**: crawl knowledge base (Zendesk/Confluence/docs), extract text, chunk, embed, store in vector DB. Run on a schedule (daily) to keep updated. (2) **Query pipeline**: user message -> embed -> retrieve top-5 chunks -> rerank -> build prompt with system instructions + context + conversation history -> stream LLM response. (3) **Conversation management**: store chat sessions in PostgreSQL, load history on reconnect. (4) **Escalation**: if confidence is low (retrieval scores below threshold) or user explicitly asks, route to human agent with conversation context. (5) **Feedback loop**: log all interactions, collect thumbs up/down, use for eval and continuous improvement. (6) **Guardrails**: don't promise refunds/policies the company doesn't have, PII handling, topic boundaries. **Scale**: Redis cache for frequent questions, multiple backend replicas, managed vector DB. This is essentially a production version of this project.

#### 47. How would you build a document processing pipeline that handles PDFs, Word docs, emails, and images?

**Ingestion layer**: (1) file type detection and routing, (2) PDF -> PyMuPDF (text) + Tesseract/AWS Textract (scanned/image PDFs), (3) Word -> python-docx, (4) Email -> email parser for body + attachment extraction, (5) Images -> OCR + vision model for description. **Normalization**: all extractors output the same schema: `{text, metadata, source, page/section}`. **Chunking**: apply the same chunking strategy regardless of source format, but preserve structure hints (headers, tables). **Tables**: detect tables in PDFs/docs, convert to markdown format before chunking — tables chunked row-by-row lose meaning. **Quality checks**: detect empty/corrupt files, language detection, text quality scoring (is it OCR garbage?). **Pipeline orchestration**: Celery/Airflow for async processing, dead letter queue for failures, retry logic, status tracking per document. This extends the current project's `DocumentService` which only handles text PDFs.

#### 48. You're building an AI feature and the PM says "make it faster." What do you do?

First, **measure** — where is the time going? Typical RAG breakdown: (1) embedding the query: 50-200ms, (2) vector search: 10-50ms, (3) LLM generation: 500-3000ms (dominates). **Optimizations by layer**: Embedding: cache frequent query embeddings, use smaller dimensions. Vector search: already fast, rarely the bottleneck. LLM: (1) **streaming** — show tokens as they arrive so perceived latency drops from 3s to 200ms (this project already does this), (2) use a faster/smaller model (gpt-4o-mini vs gpt-4o), (3) reduce input tokens (fewer chunks, shorter history, shorter system prompt), (4) parallel retrieval + LLM calls where possible. **Caching**: exact-match cache for common queries gives 0ms response. **Architecture**: pre-compute answers for top-50 frequent questions. Always measure before and after — don't optimize what isn't slow.

---

### Behavioral / Product Thinking

#### 49. An LLM feature you built is giving wrong answers 15% of the time. How do you handle this?

(1) **Categorize the failures** — look at the 15% and classify: is it retrieval failures (right answer isn't in retrieved chunks)? Hallucinations (model makes stuff up)? Ambiguous questions? Outdated data? (2) **Prioritize by impact** — wrong answers about pricing/legal are worse than wrong answers about formatting. (3) **Quick fixes**: adjust retrieval (more chunks, lower threshold), improve system prompt, add guardrails for high-risk topics. (4) **Medium-term**: build eval suite from the failure cases, add reranking, improve chunking. (5) **Communicate** — tell the PM/users the accuracy rate and where failures happen. Set expectations: 100% accuracy is impossible with LLMs. (6) **Add confidence signals** — show retrieval scores, add "I'm not sure about this" when confidence is low, always show sources so users can verify.

#### 50. Your team wants to add AI to a product. How do you decide if it's the right approach?

Ask these questions: (1) **Is the task well-defined enough?** "Summarize this document" is a good AI task. "Make the product better" is not. (2) **What's the cost of being wrong?** AI for movie recommendations (low stakes) vs AI for medical diagnosis (high stakes) need different confidence levels. (3) **Is there existing data?** RAG needs documents. Fine-tuning needs examples. No data = no AI. (4) **What's the baseline?** If keyword search already works 90% of the time, AI adding 5% improvement might not justify the cost/complexity. (5) **Can you evaluate it?** If you can't measure whether the AI is good, you can't improve it. (6) **Cost math** — at $0.005/query and 10k queries/day, that's $1,500/month. Is the feature worth it? Often the answer is: start with a simple rule-based system, add AI only where it clearly outperforms, and always have a fallback.