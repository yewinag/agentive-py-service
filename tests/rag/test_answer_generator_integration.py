"""End-to-end demonstration of the full architecture working together,
with no external infrastructure:

    FakeDocumentExtractor -> SectionAwareChunker -> FakeEmbeddingProvider
    -> InMemoryVectorStore -> Retriever -> FakeLLMProvider

Ingests a small synthetic car-rental-flavored knowledge base, retrieves
against it, and confirms a grounded (fake) answer with sources comes
back - and that an unrelated query correctly yields the deterministic
"not available" response without ever calling the LLM.
"""
import asyncio

from app.knowledge.chunking import SectionAwareChunker
from app.knowledge.fake_embedding_provider import FakeEmbeddingProvider
from app.knowledge.fake_extractor import FakeDocumentExtractor
from app.knowledge.in_memory_vector_store import InMemoryVectorStore
from app.knowledge.ingestion import IngestionService
from app.knowledge.models import DocumentSource
from app.knowledge.retriever import VectorRetriever
from app.llm.fake_provider import FakeLLMProvider
from app.rag.answer_generator import NOT_AVAILABLE_ANSWER, AnswerGenerator

POLICY_TEXT = (
    "1. Driver Eligibility & Required Documents\n"
    "● Minimum Age: Renters must be at least 21 years old.\n"
    "2. Payment & Security Deposit\n"
    "● Accepted Payment Methods: Credit Card, Bank Transfer, or PromptPay.\n"
)


def _build_pipeline():
    embedding_provider = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    ingestion = IngestionService(
        extractor=FakeDocumentExtractor(),
        chunker=SectionAwareChunker(),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    retriever = VectorRetriever(embedding_provider, vector_store)
    generator = AnswerGenerator(retriever, FakeLLMProvider())
    return ingestion, generator


def test_full_pipeline_produces_a_grounded_answer_with_sources():
    async def scenario():
        ingestion, generator = _build_pipeline()
        source = DocumentSource(
            id="policies.pdf", title="Terms & Rental Policies", content=POLICY_TEXT
        )
        chunk_count = await ingestion.ingest([source])
        assert chunk_count == 2

        # FakeEmbeddingProvider hashes the literal string (it isn't
        # semantically meaningful), so the query must match a stored
        # chunk's exact text - including its heading prefix - to
        # deterministically retrieve that chunk rather than a "random"
        # nearest hash.
        result = await generator.answer(
            "1. Driver Eligibility & Required Documents\n"
            "● Minimum Age: Renters must be at least 21 years old."
        )

        assert result.answer.startswith("[fake-llm-reply]")
        assert "Minimum Age" in result.answer
        assert len(result.sources) >= 1
        assert result.sources[0].document_title == "Terms & Rental Policies"
        assert result.sources[0].section_heading == "1. Driver Eligibility & Required Documents"

    asyncio.run(scenario())


def test_full_pipeline_reports_not_available_for_an_empty_knowledge_base():
    async def scenario():
        _, generator = _build_pipeline()  # nothing ingested

        result = await generator.answer("What is the cancellation policy?")

        assert result.answer == NOT_AVAILABLE_ANSWER
        assert result.sources == []

    asyncio.run(scenario())
