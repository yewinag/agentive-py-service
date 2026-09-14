from typing import Protocol

from app.core.config import Settings
from app.knowledge.fake_embedding_provider import FakeEmbeddingProvider
from app.knowledge.openai_embedding_provider import OpenAIEmbeddingProvider


class EmbeddingProvider(Protocol):
    """The boundary a future ingestion pipeline codes against: WHAT it
    needs to turn chunk text into vectors. Concrete providers (fake
    today, a real SDK-backed one later) implement this structurally -
    the same pattern as LLMProvider and DocumentExtractor.

    Batch-first by design: ingestion will always be embedding many
    DocumentChunks at once, never one text in isolation, so the
    contract is texts -> embeddings rather than text -> embedding.
    """

    @property
    def dimensions(self) -> int:
        """Length of every vector this provider returns. Fixed per
        provider/model - callers (e.g. a future vector store) can read
        this once to size storage, without embedding anything first.
        """
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns one embedding per input text, in the same order."""
        ...


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Composition point for EmbeddingProvider - selects the concrete
    provider from Settings.embedding_provider. Deliberately a plain
    function, not FastAPI Depends()-wired: nothing in the app consumes
    an EmbeddingProvider yet (no ingestion endpoint exists), and
    app/knowledge has no FastAPI dependency today (see README) - adding
    Depends() here ahead of a real consumer would be speculative
    plumbing. A future FastAPI-facing composition point can trivially
    wrap this one: `Depends(lambda s=Depends(get_settings): get_embedding_provider(s))`.
    """
    if settings.embedding_provider == "fake":
        return FakeEmbeddingProvider()

    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )

    raise NotImplementedError(
        f"Embedding provider '{settings.embedding_provider}' is not implemented yet"
    )
