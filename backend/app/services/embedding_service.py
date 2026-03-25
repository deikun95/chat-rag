"""
EmbeddingService — converts text to vector embeddings.

Currently uses OpenAI text-embedding-3-small.
Isolated behind an interface so we can swap to Voyage/local model later.
"""

import logging
from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
MAX_BATCH_SIZE = 100  # OpenAI allows up to 2048, but smaller batches = less memory


class EmbeddingService:
    """Converts text strings to vector embeddings."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = EMBEDDING_MODEL
        self.dimensions = EMBEDDING_DIMENSIONS

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts into vectors.

        Automatically batches if list is longer than MAX_BATCH_SIZE.
        Returns vectors in the same order as input texts.
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i : i + MAX_BATCH_SIZE]
            logger.debug(
                f"Embedding batch {i // MAX_BATCH_SIZE + 1}: "
                f"{len(batch)} texts"
            )

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            # Response items are in same order as input
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        logger.info(f"Embedded {len(texts)} texts -> {len(all_embeddings)} vectors")
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Convenience method — just calls embed_texts with a single item.
        """
        vectors = self.embed_texts([query])
        return vectors[0]
