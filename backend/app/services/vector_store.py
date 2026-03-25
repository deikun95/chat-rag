"""
VectorStoreService — abstraction over the vector database.

Currently uses ChromaDB (persistent mode).
"""

import logging
import chromadb

from app.core.config import settings

logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "document_chunks"


class VectorStoreService:
    """Stores and retrieves document chunks with vector embeddings."""

    def __init__(self):
        """Initialize ChromaDB with persistent storage."""
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        logger.info(
            f"ChromaDB initialized: {settings.chroma_persist_dir}, "
            f"collection '{COLLECTION_NAME}' "
            f"({self.collection.count()} existing chunks)"
        )

    def store_chunks(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> int:
        """
        Store document chunks with their embeddings.

        All lists must be the same length.
        Returns number of chunks stored.
        """
        if not ids:
            return 0

        assert len(ids) == len(texts) == len(embeddings) == len(metadatas), (
            f"Length mismatch: ids={len(ids)}, texts={len(texts)}, "
            f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"
        )

        # ChromaDB has a batch limit — process in chunks of 500
        batch_size = 500
        total_stored = 0

        for i in range(0, len(ids), batch_size):
            batch_end = i + batch_size
            self.collection.add(
                ids=ids[i:batch_end],
                documents=texts[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            total_stored += len(ids[i:batch_end])

        logger.info(f"Stored {total_stored} chunks in ChromaDB")
        return total_stored

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        document_ids: list[str] | None = None,
    ) -> list[dict]:
        """
        Search for similar chunks by vector similarity.

        Returns list of dicts with keys: id, text, metadata, distance, score
        Sorted by relevance (highest score first).
        """
        # Build filter
        where_filter = None
        if document_ids:
            if len(document_ids) == 1:
                where_filter = {"document_id": document_ids[0]}
            else:
                where_filter = {"document_id": {"$in": document_ids}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Unpack ChromaDB's nested result format
        chunks = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                chunks.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": distance,
                    "score": 1.0 - distance,  # cosine: lower distance = more similar
                })

        if chunks:
            logger.debug(
                f"Search returned {len(chunks)} results "
                f"(top score: {chunks[0]['score']:.3f})"
            )
        else:
            logger.debug("Search returned 0 results")
        return chunks

    def delete_by_document(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.
        Returns number of chunks deleted.
        """
        # First, find all chunk IDs for this document
        results = self.collection.get(
            where={"document_id": document_id},
            include=[],  # we only need IDs
        )

        if not results["ids"]:
            logger.info(f"No chunks found for document '{document_id}'")
            return 0

        count = len(results["ids"])
        self.collection.delete(ids=results["ids"])
        logger.info(f"Deleted {count} chunks for document '{document_id}'")
        return count

    def count(self) -> int:
        """Total number of chunks stored."""
        return self.collection.count()
