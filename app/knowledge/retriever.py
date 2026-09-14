from typing import Optional, Protocol

from app.core.config import Settings
from app.knowledge.embedding import EmbeddingProvider
from app.knowledge.vector_store import VectorSearchResult, VectorStore


class Retriever(Protocol):
    """The boundary a future RAG/chat layer codes against: WHAT it needs
    to turn a user's query into relevant, already-stored chunks. Depends
    only on EmbeddingProvider and VectorStore - never FastAPI, the
    OpenAI SDK, or SQLAlchemy directly.
    """

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[VectorSearchResult]: ...


class VectorRetriever:
    """Retriever implementation: embeds the query with an
    EmbeddingProvider, then asks a VectorStore for the most similar
    stored chunks. Pure orchestration - it holds no embedding or search
    logic of its own; query-embedding stays out of VectorStore, and
    result-ranking stays out of EmbeddingProvider.

    `top_k` is a real number (see README): every result VectorStore
    returns is passed through unchanged - fewer stored chunks than
    top_k means fewer results, never padding to reach top_k.

    `min_score` has no enforced default (None = no filtering). There is
    no empirical basis yet for a universal similarity cutoff for a given
    embedding model/knowledge base; the parameter exists so a value
    derived from real evaluation data can be set later via
    Settings.retrieval_min_score without any code change here.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        default_top_k: int = 5,
        default_min_score: Optional[float] = None,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be positive")

        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._default_top_k = default_top_k
        self._default_min_score = default_min_score

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[VectorSearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be blank")

        resolved_top_k = top_k if top_k is not None else self._default_top_k
        if resolved_top_k <= 0:
            raise ValueError("top_k must be positive")
        resolved_min_score = min_score if min_score is not None else self._default_min_score

        [query_embedding] = await self._embedding_provider.embed([query])
        results = await self._vector_store.search(query_embedding, top_k=resolved_top_k)

        if resolved_min_score is not None:
            results = [result for result in results if result.score >= resolved_min_score]

        return results


def get_retriever(settings: Settings) -> Retriever:
    """Composition point for Retriever - a plain function, not FastAPI
    Depends()-wired, for the same reason as get_embedding_provider() and
    get_vector_store(): no retrieval endpoint exists yet. Builds its
    EmbeddingProvider and VectorStore via their own composition points,
    rather than deciding providers itself - that decision stays owned in
    exactly one place each.
    """
    from app.knowledge.embedding import get_embedding_provider
    from app.knowledge.vector_store import get_vector_store

    return VectorRetriever(
        embedding_provider=get_embedding_provider(settings),
        vector_store=get_vector_store(settings),
        default_top_k=settings.retrieval_top_k,
        default_min_score=settings.retrieval_min_score,
    )
