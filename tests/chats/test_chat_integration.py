"""End-to-end proof that POST /api/v1/chat is now actually connected to
the RAG architecture, using only in-process fakes:

    FakeDocumentExtractor -> SectionAwareChunker -> FakeEmbeddingProvider
    -> InMemoryVectorStore -> Retriever -> FakeLLMProvider
    -> AnswerGenerator -> ChatService -> FastAPI /api/v1/chat

No real OpenAI call, no PostgreSQL requirement. The composed ChatService
is injected via app.dependency_overrides[get_chat_service] - the one
FastAPI-Depends()-wired seam in this chain (see app/chats/service.py) -
so the test exercises the real router/HTTP round-trip while the
underlying providers stay deterministic.
"""
import asyncio

from app.chats.service import ChatService, get_chat_service
from app.knowledge.chunking import SectionAwareChunker
from app.knowledge.fake_embedding_provider import FakeEmbeddingProvider
from app.knowledge.fake_extractor import FakeDocumentExtractor
from app.knowledge.in_memory_vector_store import InMemoryVectorStore
from app.knowledge.ingestion import IngestionService
from app.knowledge.models import DocumentSource
from app.knowledge.retriever import VectorRetriever
from app.llm.fake_provider import FakeLLMProvider
from app.main import app
from app.rag.answer_generator import NOT_AVAILABLE_ANSWER, AnswerGenerator

POLICY_TEXT = (
    "1. Driver Eligibility & Required Documents\n"
    "● Minimum Age: Renters must be at least 21 years old.\n"
)


async def _build_chat_service_with_ingested_knowledge() -> ChatService:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    ingestion = IngestionService(
        extractor=FakeDocumentExtractor(),
        chunker=SectionAwareChunker(),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    await ingestion.ingest(
        [DocumentSource(id="policies.pdf", title="Terms & Rental Policies", content=POLICY_TEXT)]
    )
    retriever = VectorRetriever(embedding_provider, vector_store)
    return ChatService(AnswerGenerator(retriever, FakeLLMProvider()))


def _build_chat_service_with_empty_knowledge() -> ChatService:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()  # nothing ingested
    retriever = VectorRetriever(embedding_provider, vector_store)
    return ChatService(AnswerGenerator(retriever, FakeLLMProvider()))


def test_chat_endpoint_returns_a_grounded_answer_from_the_full_rag_pipeline(client):
    service = asyncio.run(_build_chat_service_with_ingested_knowledge())
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        # FakeEmbeddingProvider hashes the literal string (not
        # semantically meaningful - see Step 10/11), so the query must
        # match a stored chunk's exact text to deterministically
        # retrieve it rather than "whatever's nearest by hash".
        query = (
            "1. Driver Eligibility & Required Documents\n"
            "● Minimum Age: Renters must be at least 21 years old."
        )
        response = client.post("/api/v1/chat", json={"message": query})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["reply"].startswith("[fake-llm-reply]")
    assert "Minimum Age" in body["reply"]
    assert body["sources"] == [
        {
            "document_title": "Terms & Rental Policies",
            "section_heading": "1. Driver Eligibility & Required Documents",
        }
    ]


def test_chat_endpoint_returns_not_available_when_knowledge_base_is_empty(client):
    service = _build_chat_service_with_empty_knowledge()
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        response = client.post(
            "/api/v1/chat", json={"message": "What is the cancellation policy?"}
        )
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == NOT_AVAILABLE_ANSWER
    assert body["sources"] == []
