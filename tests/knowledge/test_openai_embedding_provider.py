import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai import OpenAIError

from app.knowledge.exceptions import EmbeddingProviderError
from app.knowledge.openai_embedding_provider import OpenAIEmbeddingProvider


def _make_client(embeddings_by_index):
    """embeddings_by_index: list of (index, vector) - deliberately not
    necessarily in index order, to prove the provider places results by
    index rather than by response list order.
    """
    data = [SimpleNamespace(index=i, embedding=v) for i, v in embeddings_by_index]
    response = SimpleNamespace(data=data)
    return SimpleNamespace(embeddings=SimpleNamespace(create=AsyncMock(return_value=response)))


def test_embed_passes_batch_input_and_configured_model():
    client = _make_client([(0, [0.1, 0.2]), (1, [0.3, 0.4])])
    provider = OpenAIEmbeddingProvider(
        api_key="unused", model="text-embedding-3-small", client=client
    )

    asyncio.run(provider.embed(["first", "second"]))

    client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-3-small", input=["first", "second"]
    )


def test_embed_returns_vectors_in_input_order_even_if_response_is_reordered():
    client = _make_client([(1, [0.3, 0.4]), (0, [0.1, 0.2])])
    provider = OpenAIEmbeddingProvider(
        api_key="unused", model="text-embedding-3-small", client=client
    )

    vectors = asyncio.run(provider.embed(["first", "second"]))

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_on_empty_input_returns_empty_list_without_calling_api():
    client = _make_client([])
    provider = OpenAIEmbeddingProvider(
        api_key="unused", model="text-embedding-3-small", client=client
    )

    vectors = asyncio.run(provider.embed([]))

    assert vectors == []
    client.embeddings.create.assert_not_awaited()


def test_embed_translates_sdk_errors_into_embedding_provider_error():
    client = SimpleNamespace(
        embeddings=SimpleNamespace(create=AsyncMock(side_effect=OpenAIError("boom")))
    )
    provider = OpenAIEmbeddingProvider(
        api_key="unused", model="text-embedding-3-small", client=client
    )

    with pytest.raises(EmbeddingProviderError):
        asyncio.run(provider.embed(["hi"]))


def test_dimensions_resolved_from_known_model():
    provider = OpenAIEmbeddingProvider(
        api_key="unused", model="text-embedding-3-small", client=_make_client([])
    )

    assert provider.dimensions == 1536


def test_dimensions_can_be_overridden_explicitly():
    provider = OpenAIEmbeddingProvider(
        api_key="unused",
        model="text-embedding-3-small",
        dimensions=256,
        client=_make_client([]),
    )

    assert provider.dimensions == 256


def test_unknown_model_without_explicit_dimensions_raises_clearly():
    with pytest.raises(EmbeddingProviderError):
        OpenAIEmbeddingProvider(
            api_key="unused", model="some-future-model", client=_make_client([])
        )
