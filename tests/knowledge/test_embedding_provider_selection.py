import pytest

from app.core.config import Settings
from app.knowledge.embedding import get_embedding_provider
from app.knowledge.fake_embedding_provider import FakeEmbeddingProvider
from app.knowledge.openai_embedding_provider import OpenAIEmbeddingProvider


def test_get_embedding_provider_returns_fake_by_default():
    settings = Settings(embedding_provider="fake")

    provider = get_embedding_provider(settings)

    assert isinstance(provider, FakeEmbeddingProvider)


def test_get_embedding_provider_returns_openai_provider_when_configured():
    settings = Settings(
        embedding_provider="openai",
        openai_api_key="sk-test",
        openai_embedding_model="text-embedding-3-small",
    )

    provider = get_embedding_provider(settings)

    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.dimensions == 1536


def test_get_embedding_provider_fails_clearly_when_openai_key_missing():
    settings = Settings(embedding_provider="openai", openai_api_key=None)

    with pytest.raises(RuntimeError):
        get_embedding_provider(settings)


def test_get_embedding_provider_rejects_unknown_provider():
    settings = Settings(embedding_provider="not-a-real-provider")

    with pytest.raises(NotImplementedError):
        get_embedding_provider(settings)
