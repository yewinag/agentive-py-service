import asyncio

from app.knowledge.fake_embedding_provider import FakeEmbeddingProvider


def test_embed_returns_one_vector_per_input_in_order():
    provider = FakeEmbeddingProvider()

    vectors = asyncio.run(provider.embed(["hello", "world"]))

    assert len(vectors) == 2
    assert all(len(v) == provider.dimensions for v in vectors)


def test_embed_is_deterministic_and_distinguishes_different_texts():
    provider = FakeEmbeddingProvider()

    first_run = asyncio.run(provider.embed(["a", "b"]))
    second_run = asyncio.run(provider.embed(["a", "b"]))

    assert first_run == second_run
    assert first_run[0] != first_run[1]


def test_dimensions_is_configurable():
    provider = FakeEmbeddingProvider(dimensions=16)

    vectors = asyncio.run(provider.embed(["hello"]))

    assert provider.dimensions == 16
    assert len(vectors[0]) == 16
