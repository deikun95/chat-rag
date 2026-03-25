"""
RetrievalService — finds the most relevant chunks for a query.

Pipeline: embed query -> vector search -> return top-k
"""

import logging

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class RetrievalService:
    """Retrieves relevant document chunks for a given query."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find the most relevant chunks for a query.

        Steps:
        1. Embed the query using the same model as documents
        2. Search vector store for similar chunks
        3. Filter by relevance score threshold
        4. Return top-k results
        """
        # Step 1: Embed the query
        query_embedding = self.embedding_service.embed_query(query)

        # Step 2: Vector search (get more than we need for filtering)
        search_k = min(top_k * 2, 20)
        candidates = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=search_k,
            document_ids=document_ids,
        )

        if not candidates:
            logger.info(f"No results found for query: '{query[:50]}...'")
            return []

        # Step 3: Filter by minimum relevance score
        min_score = 0.05
        filtered = [c for c in candidates if c["score"] >= min_score]

        if not filtered:
            logger.info(
                f"All {len(candidates)} results below relevance threshold "
                f"(best: {candidates[0]['score']:.3f})"
            )
            return []

        # Step 4: Take top-k
        results = filtered[:top_k]

        logger.info(
            f"Retrieved {len(results)} chunks for query "
            f"(scores: {results[0]['score']:.3f} - {results[-1]['score']:.3f})"
        )
        return results
