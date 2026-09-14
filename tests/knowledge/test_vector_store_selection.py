import pytest

from app.core.config import Settings
from app.knowledge.in_memory_vector_store import InMemoryVectorStore
from app.knowledge.pgvector_store import PgVectorStore
from app.knowledge.vector_store import get_vector_store


def test_get_vector_store_returns_in_memory_by_default():
    settings = Settings(vector_store_provider="memory")

    store = get_vector_store(settings)

    assert isinstance(store, InMemoryVectorStore)


def test_get_vector_store_returns_pgvector_store_when_configured():
    settings = Settings(
        vector_store_provider="pgvector",
        database_url="postgresql+asyncpg://user:pass@localhost/db",
        openai_embedding_model="text-embedding-3-small",
    )

    store = get_vector_store(settings)

    assert isinstance(store, PgVectorStore)


def test_get_vector_store_fails_clearly_when_database_url_missing():
    settings = Settings(vector_store_provider="pgvector", database_url=None)

    with pytest.raises(RuntimeError):
        get_vector_store(settings)


def test_get_vector_store_fails_clearly_for_unknown_embedding_model_dimensions():
    settings = Settings(
        vector_store_provider="pgvector",
        database_url="postgresql+asyncpg://user:pass@localhost/db",
        openai_embedding_model="some-future-model",
    )

    with pytest.raises(RuntimeError):
        get_vector_store(settings)


def test_get_vector_store_rejects_unknown_provider():
    settings = Settings(vector_store_provider="not-a-real-store")

    with pytest.raises(NotImplementedError):
        get_vector_store(settings)
