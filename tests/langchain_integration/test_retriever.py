"""Unit coverage for KnowledgeBaseRetriever's own logic - the
Document-translation boundary and top_k forwarding - against a stub
Retriever, needing no real Qdrant or OpenAI call. The real round trip
against the actual collection lives in
test_retriever_integration.py (skipped unless QDRANT_URL and
OPENAI_API_KEY are both set).
"""
import asyncio
from typing import Optional

import pytest
from langchain_core.documents import Document

from app.knowledge.models import DocumentChunk
from app.knowledge.vector_store import VectorSearchResult
from app.langchain_integration.retriever import KnowledgeBaseRetriever


class StubRetriever:
    """Satisfies the shape KnowledgeBaseRetriever depends on
    (Retriever.retrieve), unrelated to any real embedding/Qdrant call -
    proves the adapter's own translation logic in isolation.
    """

    def __init__(self, results: list[VectorSearchResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, Optional[int]]] = []

    async def retrieve(self, query: str, top_k: Optional[int] = None, min_score: Optional[float] = None):
        self.calls.append((query, top_k))
        return self._results


def _result(chunk_id: str, text: str, score: float, section_heading=None) -> VectorSearchResult:
    chunk = DocumentChunk(
        id=chunk_id,
        document_id="04-cancellation-policy.pdf",
        document_title="Cancellation Policy",
        section_heading=section_heading,
        text=text,
        position=0,
    )
    return VectorSearchResult(chunk=chunk, score=score)


def test_ainvoke_returns_langchain_documents_with_full_metadata():
    stub = StubRetriever([_result("c1", "50% cancellation fee applies.", 0.9, "1. Cancellation Terms")])
    retriever = KnowledgeBaseRetriever(retriever=stub)

    documents = asyncio.run(retriever.ainvoke("cancel less than 24 hours before pickup"))

    assert len(documents) == 1
    document = documents[0]
    assert isinstance(document, Document)
    assert document.page_content == "50% cancellation fee applies."
    assert document.metadata == {
        "chunk_id": "c1",
        "document_id": "04-cancellation-policy.pdf",
        "document_title": "Cancellation Policy",
        "section_heading": "1. Cancellation Terms",
        "position": 0,
        "score": 0.9,
    }


def test_ainvoke_returns_empty_list_when_nothing_is_retrieved():
    stub = StubRetriever([])
    retriever = KnowledgeBaseRetriever(retriever=stub)

    documents = asyncio.run(retriever.ainvoke("irrelevant question"))

    assert documents == []


def test_ainvoke_preserves_result_order():
    stub = StubRetriever(
        [_result("c1", "first", 0.9), _result("c2", "second", 0.8), _result("c3", "third", 0.7)]
    )
    retriever = KnowledgeBaseRetriever(retriever=stub)

    documents = asyncio.run(retriever.ainvoke("query"))

    assert [d.page_content for d in documents] == ["first", "second", "third"]


def test_top_k_is_forwarded_to_the_wrapped_retriever():
    stub = StubRetriever([])
    retriever = KnowledgeBaseRetriever(retriever=stub, top_k=3)

    asyncio.run(retriever.ainvoke("query"))

    assert stub.calls == [("query", 3)]


def test_top_k_defaults_to_none_leaving_the_wrapped_retrievers_own_default_in_effect():
    stub = StubRetriever([])
    retriever = KnowledgeBaseRetriever(retriever=stub)

    asyncio.run(retriever.ainvoke("query"))

    assert stub.calls == [("query", None)]


def test_sync_invoke_raises_not_implemented_since_the_wrapped_retriever_is_async_only():
    stub = StubRetriever([])
    retriever = KnowledgeBaseRetriever(retriever=stub)

    with pytest.raises(NotImplementedError):
        retriever.invoke("query")
