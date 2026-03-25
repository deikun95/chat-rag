"""
IngestionService — orchestrates the full document ingestion pipeline:
pages -> chunks -> embeddings -> vector store.

This is the ONLY service that knows the full ingestion flow.
"""

import logging
from dataclasses import dataclass

from app.services.document_service import PageContent
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

# ==================== Chunking config ====================

CHUNK_SIZE = 2000         # characters (~500 tokens)
CHUNK_OVERLAP = 200       # characters (~50 tokens)
SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    """A single chunk of text with metadata."""
    text: str
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int


class IngestionService:
    """Orchestrates: pages -> chunks -> embeddings -> vector store."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    # ==================== Chunking ====================

    def _recursive_split(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: list[str] | None = None,
    ) -> list[str]:
        """
        Split text recursively using a hierarchy of separators.

        Strategy:
        1. Try to split by double newline (paragraph breaks)
        2. If chunks are still too big, split by single newline
        3. Then by sentence (". ")
        4. Last resort: by space

        Overlap ensures context is preserved at chunk boundaries.
        """
        if separators is None:
            separators = SEPARATORS

        # Base case: text fits in one chunk
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        # Try each separator
        for sep in separators:
            if sep not in text:
                continue

            parts = text.split(sep)
            chunks = []
            current = ""

            for part in parts:
                # Would adding this part exceed the chunk size?
                candidate = current + sep + part if current else part

                if len(candidate) > chunk_size and current:
                    # Current chunk is full — save it
                    chunks.append(current.strip())

                    # Start new chunk with overlap from end of previous
                    if chunk_overlap > 0 and len(current) > chunk_overlap:
                        overlap = current[-chunk_overlap:]
                        current = overlap + sep + part
                    else:
                        current = part
                else:
                    current = candidate

            # Don't forget the last accumulated chunk
            if current.strip():
                chunks.append(current.strip())

            # Only return if we actually split into multiple chunks
            if len(chunks) > 1:
                return chunks
            # If we got only 1 chunk, try next separator

        # Fallback: hard split by character count
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def _create_chunks(
        self,
        pages: list[PageContent],
        document_id: str,
        document_name: str,
    ) -> list[Chunk]:
        """
        Convert extracted pages into chunks with metadata.

        Each chunk gets a metadata header prepended:
        "[Source: document_name, Page N]"
        """
        all_chunks = []
        chunk_index = 0

        for page in pages:
            # Split this page's text into chunks
            text_chunks = self._recursive_split(page.text)

            for chunk_text in text_chunks:
                # Prepend source metadata into the chunk text itself
                enriched_text = (
                    f"[Source: {document_name}, Page {page.page_number}]\n"
                    f"{chunk_text}"
                )

                all_chunks.append(Chunk(
                    text=enriched_text,
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

        logger.info(
            f"Created {len(all_chunks)} chunks from "
            f"{len(pages)} pages of '{document_name}'"
        )
        return all_chunks

    # ==================== Full Pipeline ====================

    def ingest(
        self,
        pages: list[PageContent],
        document_id: str,
        document_name: str,
    ) -> int:
        """
        Full ingestion pipeline:
        1. Split pages into chunks
        2. Generate embeddings for all chunks
        3. Store chunks + embeddings in vector database

        Returns the number of chunks stored.
        """
        # Step 1: Chunk
        chunks = self._create_chunks(pages, document_id, document_name)
        if not chunks:
            logger.warning(f"No chunks created for document '{document_id}'")
            return 0

        # Step 2: Embed
        texts_to_embed = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts_to_embed)} chunks...")
        embeddings = self.embedding_service.embed_texts(texts_to_embed)

        # Step 3: Store
        ids = [f"{c.document_id}_{c.chunk_index}" for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "document_name": c.document_name,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        stored_count = self.vector_store.store_chunks(
            ids=ids,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            f"Ingestion complete for '{document_name}': "
            f"{stored_count} chunks stored"
        )
        return stored_count
