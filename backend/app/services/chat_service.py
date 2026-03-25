"""
ChatService — orchestrates the RAG query pipeline:
1. Retrieve relevant chunks via RetrievalService
2. Build a prompt with context + question
3. Stream the LLM response via OpenAI API
4. Return sources used
"""

import json
import logging
from typing import AsyncGenerator

from openai import OpenAI

from app.core.config import settings
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

# ==================== Prompt ====================

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided document excerpts.

RULES:
1. Answer ONLY based on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to fully answer the question, do your best with what is available. Briefly explain what the documents DO cover, and mention that the specific information the user asked about was not found. For example: "The uploaded documents cover [topic X], but I couldn't find specific information about [user's question]. Here's what I did find that might be relevant: ..."
3. Cite your sources using [Source N] format, where N matches the source number in the context.
4. Be concise and direct. Get to the point quickly.
5. If sources contradict each other, mention the contradiction.
6. Preserve technical accuracy — do not round numbers, change dates, or simplify specific terms.
7. Use the same language as the user's question."""

MAX_HISTORY_MESSAGES = 20  # keep last N messages to avoid token overflow


class ChatService:
    """Orchestrates RAG: retrieve context -> build prompt -> stream LLM."""

    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval_service = retrieval_service
        self.client = OpenAI(api_key=settings.openai_api_key)

    def _build_context(self, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a numbered context block
        that the LLM can reference with [Source N] citations.
        """
        if not chunks:
            return "No relevant context found in the uploaded documents."

        parts = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk["metadata"]
            parts.append(
                f"[Source {i}] "
                f"(Document: {meta['document_name']}, "
                f"Page {meta['page_number']})\n"
                f"{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def _build_user_message(self, context: str, question: str) -> str:
        """Build the user message with context and question."""
        return f"""Here are relevant excerpts from the uploaded documents:

---
{context}
---

Based on the above context, please answer the following question:
{question}"""

    async def stream_response(
        self,
        question: str,
        document_ids: list[str] | None = None,
        history: list[dict] | None = None,
        top_k: int = 5,
    ) -> AsyncGenerator[str, None]:
        """
        Full RAG pipeline with SSE streaming.

        Yields Server-Sent Event formatted strings:
        - "event: chunk\ndata: {...}\n\n"  — text tokens
        - "event: sources\ndata: {...}\n\n" — source references
        - "event: done\ndata: {...}\n\n"    — completion signal
        """
        # Step 1: Retrieve relevant chunks
        logger.info(f"Processing question: '{question[:80]}...'")
        chunks = self.retrieval_service.retrieve(
            query=question,
            document_ids=document_ids,
            top_k=top_k,
        )

        # Step 2: Build messages with history
        context = self._build_context(chunks)
        user_message = self._build_user_message(context, question)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history (truncated to last N)
        if history:
            for msg in history[-MAX_HISTORY_MESSAGES:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        messages.append({"role": "user", "content": user_message})

        logger.info(
            f"Sending {len(messages)} messages to LLM "
            f"({len(history or [])} history)"
        )

        # Step 3: Stream LLM response
        try:
            stream = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=2048,
                stream=True,
                messages=messages,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    event_data = json.dumps({
                        "type": "text",
                        "content": delta.content,
                    })
                    yield f"event: chunk\ndata: {event_data}\n\n"

        except Exception as e:
            logger.error(f"LLM streaming error: {e}", exc_info=True)
            error_data = json.dumps({
                "type": "error",
                "message": "Failed to generate response. Please try again.",
            })
            yield f"event: error\ndata: {error_data}\n\n"
            return

        # Step 4: Send source references
        sources = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk["metadata"]
            sources.append({
                "index": i,
                "document_id": meta["document_id"],
                "document_name": meta["document_name"],
                "page": meta["page_number"],
                "chunk_text": chunk["text"][:300],
                "relevance_score": round(chunk["score"], 3),
            })

        sources_data = json.dumps({"sources": sources})
        yield f"event: sources\ndata: {sources_data}\n\n"

        # Step 5: Done signal
        done_data = json.dumps({"status": "complete"})
        yield f"event: done\ndata: {done_data}\n\n"
