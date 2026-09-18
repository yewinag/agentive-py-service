"""Unit coverage for LangChainRagGenerationService's own logic - context
building, source extraction, and the no-context short-circuit - against
a stub retriever and LangChain's own FakeListChatModel. No real Qdrant
or OpenAI call, the same "fake provider" pattern this project already
uses for its own LLMProvider/EmbeddingProvider. The real round trip
against a real chat model lives in test_generation_integration.py
(skipped unless OPENAI_API_KEY is set, and deliberately only a handful
of real calls - see that file's docstring for the cost-awareness note).
"""
import asyncio

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.langchain_integration.generation import (
    NOT_AVAILABLE_ANSWER,
    LangChainRagGenerationService,
    RagAnswer,
    RagSource,
    _build_context,
)


class StubKnowledgeBaseRetriever:
    """Satisfies the shape LangChainRagGenerationService depends on
    (KnowledgeBaseRetriever.ainvoke), unrelated to any real
    embedding/Qdrant call - proves the generation service's own prompt
    building/source-extraction logic in isolation.
    """

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self.calls: list[str] = []

    async def ainvoke(self, query: str) -> list[Document]:
        self.calls.append(query)
        return self._documents


def _document(text: str, document_title: str, section_heading=None) -> Document:
    return Document(
        page_content=text,
        metadata={
            "chunk_id": "c1",
            "document_id": "04-cancellation-policy.pdf",
            "document_title": document_title,
            "section_heading": section_heading,
            "position": 0,
            "score": 0.9,
        },
    )


def test_agenerate_returns_the_chat_models_answer_and_sources_from_retrieved_documents():
    documents = [
        _document("50% cancellation fee applies.", "Cancellation Policy", "1. Cancellation Terms")
    ]
    retriever = StubKnowledgeBaseRetriever(documents)
    chat_model = FakeListChatModel(responses=["A 50% fee applies for cancellations under 24 hours."])
    service = LangChainRagGenerationService(retriever=retriever, chat_model=chat_model)

    result = asyncio.run(service.agenerate("What happens if I cancel late?"))

    assert isinstance(result, RagAnswer)
    assert result.answer == "A 50% fee applies for cancellations under 24 hours."
    assert result.sources == [RagSource(document_title="Cancellation Policy", section_heading="1. Cancellation Terms")]
    assert retriever.calls == ["What happens if I cancel late?"]


def test_agenerate_never_calls_the_chat_model_when_nothing_is_retrieved():
    retriever = StubKnowledgeBaseRetriever([])
    chat_model = FakeListChatModel(responses=["this should never be returned"])
    service = LangChainRagGenerationService(retriever=retriever, chat_model=chat_model)

    result = asyncio.run(service.agenerate("Do you offer free baby seats?"))

    assert result.answer == NOT_AVAILABLE_ANSWER
    assert result.sources == []


def test_agenerate_includes_multiple_sources_in_order():
    documents = [
        _document("first chunk text", "Cancellation Policy", "1. Cancellation Terms"),
        _document("second chunk text", "Payment Policy", "2. Security Deposit"),
    ]
    retriever = StubKnowledgeBaseRetriever(documents)
    chat_model = FakeListChatModel(responses=["combined answer"])
    service = LangChainRagGenerationService(retriever=retriever, chat_model=chat_model)

    result = asyncio.run(service.agenerate("question"))

    assert [s.document_title for s in result.sources] == ["Cancellation Policy", "Payment Policy"]


def test_build_context_includes_document_title_heading_and_text():
    documents = [
        _document("Exact policy text.", "Cancellation Policy", "1. Cancellation Terms"),
        _document("No heading here.", "Rental Services", None),
    ]

    context = _build_context(documents)

    assert "Source: Cancellation Policy — 1. Cancellation Terms" in context
    assert "Exact policy text." in context
    assert "Source: Rental Services\nNo heading here." in context
