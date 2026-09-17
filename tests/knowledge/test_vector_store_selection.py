import pytest

from app.core.config import Settings
from app.knowledge.exceptions import EmbeddingProviderError
from app.knowledge.fake_embedding_provider import FakeEmbeddingProvider
from app.knowledge.in_memory_vector_store import InMemoryVectorStore
from app.knowledge.pgvector_store import PgVectorStore
from app.knowledge.qdrant_vector_store import QdrantVectorStore
from app.knowledge.vector_store import get_vector_store, reset_default_vector_store


@pytest.fixture(autouse=True)
def _reset_in_memory_singleton():
    reset_default_vector_store()
    yield
    reset_default_vector_store()


def test_get_vector_store_returns_in_memory_by_default():
    settings = Settings(vector_store_provider="memory")

    store = get_vector_store(settings)

    assert isinstance(store, InMemoryVectorStore)


def test_get_vector_store_returns_the_same_in_memory_instance_across_calls():
    settings = Settings(vector_store_provider="memory")

    first = get_vector_store(settings)
    second = get_vector_store(settings)

    assert first is second


def test_reset_default_vector_store_clears_the_singleton():
    settings = Settings(vector_store_provider="memory")
    first = get_vector_store(settings)

    reset_default_vector_store()
    second = get_vector_store(settings)

    assert first is not second


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


def test_get_vector_store_fails_clearly_for_unknown_openai_embedding_model_dimensions():
    settings = Settings(
        vector_store_provider="pgvector",
        database_url="postgresql+asyncpg://user:pass@localhost/db",
        embedding_provider="openai",
        openai_api_key="sk-test",
        openai_embedding_model="some-future-model",
    )

    with pytest.raises(EmbeddingProviderError):
        get_vector_store(settings)


def test_get_vector_store_returns_qdrant_store_when_configured():
    settings = Settings(
        vector_store_provider="qdrant",
        qdrant_url="http://localhost:6333",
        qdrant_collection="test_collection",
        openai_embedding_model="text-embedding-3-small",
    )

    store = get_vector_store(settings)

    assert isinstance(store, QdrantVectorStore)


def test_get_vector_store_sizes_pgvector_for_the_configured_fake_embedding_provider():
    """embedding_provider defaults to "fake" - the store must be sized
    for FakeEmbeddingProvider's actual output width (8), not whatever
    openai_embedding_model happens to say, since no OpenAI embedding
    will ever be written to this store while embedding_provider=fake.
    """
    settings = Settings(
        vector_store_provider="pgvector",
        database_url="postgresql+asyncpg://user:pass@localhost/db",
    )

    store = get_vector_store(settings)

    assert store.dimensions == FakeEmbeddingProvider().dimensions


def test_get_vector_store_sizes_qdrant_for_the_configured_fake_embedding_provider():
    settings = Settings(vector_store_provider="qdrant")

    store = get_vector_store(settings)

    assert store.dimensions == FakeEmbeddingProvider().dimensions


def test_get_vector_store_sizes_pgvector_and_qdrant_identically_for_the_same_embedding_provider():
    """Both stores derive dimensions from the same EmbeddingProvider
    instance's own declared width, so switching VECTOR_STORE_PROVIDER
    alone (embedding_provider/model unchanged) can never change the
    resolved vector width.
    """
    common = {
        "embedding_provider": "openai",
        "openai_api_key": "sk-test",
        "openai_embedding_model": "text-embedding-3-small",
    }

    pgvector_store = get_vector_store(
        Settings(vector_store_provider="pgvector", database_url="postgresql+asyncpg://user:pass@localhost/db", **common)
    )
    qdrant_store = get_vector_store(Settings(vector_store_provider="qdrant", **common))

    assert pgvector_store.dimensions == qdrant_store.dimensions == 1536


def test_get_vector_store_rejects_unknown_provider():
    settings = Settings(vector_store_provider="not-a-real-store")

    with pytest.raises(NotImplementedError):
        get_vector_store(settings)
