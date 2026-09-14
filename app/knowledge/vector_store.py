from typing import Protocol

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.knowledge.models import DocumentChunk


class VectorRecord(BaseModel):
    """A DocumentChunk paired with its embedding, as written to a
    VectorStore. Deliberately not a field added onto DocumentChunk
    itself: the embedding is a property of how a chunk is stored/indexed
    for search, not a property of the chunk as extracted/chunked - a
    chunk can exist (and be re-embedded by a different model) without
    ever having been stored.
    """

    chunk: DocumentChunk
    embedding: list[float]


class VectorSearchResult(BaseModel):
    """One ranked hit from VectorStore.search(). Carries the stored
    chunk (already has document_id, document_title, section_heading,
    text - everything a future RAG layer needs to cite and use it) plus
    a similarity score. Deliberately excludes the vector itself: a
    caller that just retrieved a chunk by similarity has no use for
    seeing its own query echoed back as a number list.
    """

    chunk: DocumentChunk
    score: float = Field(..., description="Cosine similarity; 1.0 = identical, higher = more similar.")


class VectorStore(Protocol):
    """The boundary a future retrieval layer codes against: WHAT it
    needs to persist embedded chunks and find the most similar ones to a
    query. Concrete stores (in-memory today, pgvector-backed later)
    implement this structurally - the same pattern as LLMProvider,
    DocumentExtractor, and EmbeddingProvider.

    Deliberately not generic CRUD: `add` is upsert-by-chunk-id (batch,
    since ingestion always writes many chunks at once) and `search` is
    the one read operation retrieval actually needs - there is no
    get/update/delete because nothing in this codebase has a reason to
    fetch or mutate a single stored chunk in isolation yet.
    """

    async def add(self, records: list[VectorRecord]) -> None:
        """Stores records, replacing any existing record with the same
        chunk id (idempotent: re-ingesting the same document doesn't
        duplicate its chunks).
        """
        ...

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[VectorSearchResult]:
        """Returns up to top_k stored records most similar to
        query_embedding, ordered by descending similarity.
        """
        ...


def get_vector_store(settings: Settings) -> VectorStore:
    """Composition point for VectorStore - selects the concrete store
    from Settings.vector_store_provider. A plain function, not FastAPI
    Depends()-wired, for the same reason as get_embedding_provider(): no
    ingestion or retrieval endpoint consumes a VectorStore yet, and
    app/knowledge has no FastAPI dependency today.
    """
    from app.knowledge.in_memory_vector_store import InMemoryVectorStore
    from app.knowledge.pgvector_store import PgVectorStore

    if settings.vector_store_provider == "memory":
        return InMemoryVectorStore()

    if settings.vector_store_provider == "pgvector":
        if not settings.database_url:
            raise RuntimeError(
                "DATABASE_URL is required when VECTOR_STORE_PROVIDER=pgvector"
            )
        return PgVectorStore(
            database_url=settings.database_url,
            dimensions=_embedding_dimensions_for(settings),
        )

    raise NotImplementedError(
        f"Vector store '{settings.vector_store_provider}' is not implemented yet"
    )


def _embedding_dimensions_for(settings: Settings) -> int:
    """PgVectorStore's table needs a fixed vector width up front. Reuses
    the same known-model lookup EmbeddingProvider already uses, so the
    two never drift apart, rather than a separately configured number.
    """
    from app.knowledge.openai_embedding_provider import known_embedding_dimensions

    dimensions = known_embedding_dimensions(settings.openai_embedding_model)
    if dimensions is None:
        raise RuntimeError(
            f"Unknown output dimensions for embedding model "
            f"'{settings.openai_embedding_model}'; cannot size the pgvector table."
        )
    return dimensions
