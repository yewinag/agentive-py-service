import asyncio

import pytest

from app.knowledge.exceptions import EmbeddingProviderError, VectorStoreError
from app.knowledge.models import DocumentChunk
from app.knowledge.retriever import VectorRetriever
from app.knowledge.vector_store import VectorSearchResult


class StubEmbeddingProvider:
    """Records every embed() call so tests can assert exactly what text
    the Retriever sent, without depending on a real/mocked OpenAI SDK
    (already covered by Step 8's tests).
    """

    def __init__(self, vector=None, error=None):
        self.calls = []
        self._vector = vector if vector is not None else [1.0, 0.0]
        self._error = error

    @property
    def dimensions(self) -> int:
        return len(self._vector)

    async def embed(self, texts):
        self.calls.append(list(texts))
        if self._error:
            raise self._error
        return [self._vector for _ in texts]


class StubVectorStore:
    """Records every search() call so tests can assert exactly what
    embedding/top_k the Retriever forwarded, without depending on a
    real/mocked pgvector store (already covered by Step 9's tests).
    """

    def __init__(self, results=None, error=None):
        self.search_calls = []
        self._results = results if results is not None else []
        self._error = error

    async def add(self, records):
        raise NotImplementedError("Retriever must not write to VectorStore")

    async def search(self, query_embedding, top_k=5):
        self.search_calls.append((query_embedding, top_k))
        if self._error:
            raise self._error
        return self._results


def _chunk(chunk_id: str, score: float) -> VectorSearchResult:
    return VectorSearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            document_id="doc-1",
            document_title="Doc",
            section_heading="1. Section",
            text=f"text for {chunk_id}",
            position=0,
        ),
        score=score,
    )


def test_retrieve_sends_query_text_to_embedding_provider():
    embedding_provider = StubEmbeddingProvider()
    retriever = VectorRetriever(embedding_provider, StubVectorStore())

    asyncio.run(retriever.retrieve("what is the deposit"))

    assert embedding_provider.calls == [["what is the deposit"]]


def test_retrieve_passes_resulting_embedding_to_vector_store():
    embedding_provider = StubEmbeddingProvider(vector=[0.1, 0.2, 0.3])
    vector_store = StubVectorStore()
    retriever = VectorRetriever(embedding_provider, vector_store)

    asyncio.run(retriever.retrieve("query"))

    (query_embedding, _top_k), = vector_store.search_calls
    assert query_embedding == [0.1, 0.2, 0.3]


def test_retrieve_forwards_explicit_top_k():
    vector_store = StubVectorStore()
    retriever = VectorRetriever(StubEmbeddingProvider(), vector_store, default_top_k=5)

    asyncio.run(retriever.retrieve("query", top_k=2))

    (_embedding, top_k), = vector_store.search_calls
    assert top_k == 2


def test_retrieve_uses_default_top_k_when_not_specified():
    vector_store = StubVectorStore()
    retriever = VectorRetriever(StubEmbeddingProvider(), vector_store, default_top_k=7)

    asyncio.run(retriever.retrieve("query"))

    (_embedding, top_k), = vector_store.search_calls
    assert top_k == 7


def test_retrieve_returns_vector_search_results_unchanged():
    results = [_chunk("a", 0.9), _chunk("b", 0.5)]
    retriever = VectorRetriever(StubEmbeddingProvider(), StubVectorStore(results=results))

    returned = asyncio.run(retriever.retrieve("query"))

    assert returned == results


def test_retrieve_does_not_pad_when_store_returns_fewer_than_top_k():
    results = [_chunk("only-one", 0.8)]
    retriever = VectorRetriever(
        StubEmbeddingProvider(), StubVectorStore(results=results), default_top_k=10
    )

    returned = asyncio.run(retriever.retrieve("query", top_k=10))

    assert len(returned) == 1


def test_retrieve_returns_empty_list_when_store_has_no_matches():
    retriever = VectorRetriever(StubEmbeddingProvider(), StubVectorStore(results=[]))

    returned = asyncio.run(retriever.retrieve("query"))

    assert returned == []


def test_retrieve_propagates_embedding_provider_errors_unwrapped():
    embedding_provider = StubEmbeddingProvider(error=EmbeddingProviderError("boom"))
    retriever = VectorRetriever(embedding_provider, StubVectorStore())

    with pytest.raises(EmbeddingProviderError):
        asyncio.run(retriever.retrieve("query"))


def test_retrieve_propagates_vector_store_errors_unwrapped():
    vector_store = StubVectorStore(error=VectorStoreError("boom"))
    retriever = VectorRetriever(StubEmbeddingProvider(), vector_store)

    with pytest.raises(VectorStoreError):
        asyncio.run(retriever.retrieve("query"))


def test_retrieve_rejects_blank_query():
    retriever = VectorRetriever(StubEmbeddingProvider(), StubVectorStore())

    with pytest.raises(ValueError):
        asyncio.run(retriever.retrieve("   "))


def test_retrieve_rejects_non_positive_top_k():
    retriever = VectorRetriever(StubEmbeddingProvider(), StubVectorStore())

    with pytest.raises(ValueError):
        asyncio.run(retriever.retrieve("query", top_k=0))


def test_constructor_rejects_non_positive_default_top_k():
    with pytest.raises(ValueError):
        VectorRetriever(StubEmbeddingProvider(), StubVectorStore(), default_top_k=0)


def test_min_score_filters_out_results_below_threshold():
    results = [_chunk("high", 0.95), _chunk("low", 0.4)]
    retriever = VectorRetriever(StubEmbeddingProvider(), StubVectorStore(results=results))

    returned = asyncio.run(retriever.retrieve("query", min_score=0.8))

    assert [r.chunk.id for r in returned] == ["high"]


def test_min_score_not_applied_by_default():
    results = [_chunk("high", 0.95), _chunk("low", 0.01)]
    retriever = VectorRetriever(StubEmbeddingProvider(), StubVectorStore(results=results))

    returned = asyncio.run(retriever.retrieve("query"))

    assert [r.chunk.id for r in returned] == ["high", "low"]


def test_default_min_score_applies_when_not_overridden_per_call():
    results = [_chunk("high", 0.95), _chunk("low", 0.4)]
    retriever = VectorRetriever(
        StubEmbeddingProvider(), StubVectorStore(results=results), default_min_score=0.8
    )

    returned = asyncio.run(retriever.retrieve("query"))

    assert [r.chunk.id for r in returned] == ["high"]
