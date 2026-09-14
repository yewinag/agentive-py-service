"""End-to-end proof of tool-assisted answering with no external
infrastructure:

    User question -> AgentService -> LLM decides to call the
    availability tool -> CheckVehicleAvailabilityTool executes against
    FakeBusinessServiceClient -> result goes back to the LLM -> final
    answer, grounded in both the knowledge base and the tool result.

Uses the real Retriever/InMemoryVectorStore/FakeEmbeddingProvider (via
IngestionService, exactly as Step 8-10 built them), the real
CheckVehicleAvailabilityTool/ToolRegistry (Step 14), and a scripted
FakeLLMProvider sequence (never "smart" - see README). No real OpenAI
call, no NestJS dependency, no database.
"""
import asyncio
from datetime import datetime, timedelta

from app.agent.service import AgentService
from app.knowledge.chunking import SectionAwareChunker
from app.knowledge.fake_embedding_provider import FakeEmbeddingProvider
from app.knowledge.fake_extractor import FakeDocumentExtractor
from app.knowledge.in_memory_vector_store import InMemoryVectorStore
from app.knowledge.ingestion import IngestionService
from app.knowledge.models import DocumentSource
from app.knowledge.retriever import VectorRetriever
from app.llm.fake_provider import FakeLLMProvider
from app.llm.models import LLMResponse, ToolCall
from app.tools.business_client import VehicleAvailability
from app.tools.check_vehicle_availability import TOOL_NAME, CheckVehicleAvailabilityTool
from app.tools.fake_business_client import FakeBusinessServiceClient
from app.tools.registry import ToolRegistry

POLICY_TEXT = (
    "1. Driver Eligibility & Required Documents\n"
    "● Minimum Age: Renters must be at least 21 years old.\n"
)

NOW = datetime(2026, 6, 1, 10, 0)


def test_agent_calls_the_availability_tool_and_produces_a_grounded_final_answer():
    async def scenario():
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

        vehicle = VehicleAvailability(
            vehicle_id="v1",
            category="SUV",
            model="Honda CR-V",
            available_from=NOW - timedelta(days=1),
            available_until=NOW + timedelta(days=14),
        )
        business_client = FakeBusinessServiceClient(vehicles=[vehicle])
        tool_registry = ToolRegistry()
        tool_registry.register(CheckVehicleAvailabilityTool(business_client))

        tool_call = ToolCall(
            id="call-1",
            tool_name=TOOL_NAME,
            arguments={
                "pickup_at": NOW.isoformat(),
                "return_at": (NOW + timedelta(days=2)).isoformat(),
                "category": "SUV",
            },
        )
        llm = FakeLLMProvider(
            responses=[
                LLMResponse(tool_calls=[tool_call]),
                LLMResponse(text="Yes, a Honda CR-V (SUV) is available for your dates."),
            ]
        )

        agent = AgentService(retriever, llm, tool_registry)

        # The query text matches a stored knowledge chunk exactly
        # (FakeEmbeddingProvider hashes the literal string - see Step
        # 10/11), so retrieval also succeeds and contributes sources,
        # demonstrating RAG and tools working together in one request.
        query = (
            "1. Driver Eligibility & Required Documents\n"
            "● Minimum Age: Renters must be at least 21 years old."
        )
        return await agent.answer(query)

    result = asyncio.run(scenario())

    assert result.answer == "Yes, a Honda CR-V (SUV) is available for your dates."
    assert result.sources[0].document_title == "Terms & Rental Policies"


def test_agent_gracefully_reports_no_availability():
    async def scenario():
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

        business_client = FakeBusinessServiceClient(vehicles=[])  # nothing available
        tool_registry = ToolRegistry()
        tool_registry.register(CheckVehicleAvailabilityTool(business_client))

        tool_call = ToolCall(
            id="call-1",
            tool_name=TOOL_NAME,
            arguments={
                "pickup_at": NOW.isoformat(),
                "return_at": (NOW + timedelta(days=2)).isoformat(),
            },
        )
        llm = FakeLLMProvider(
            responses=[
                LLMResponse(tool_calls=[tool_call]),
                LLMResponse(text="Unfortunately, nothing is available for those dates."),
            ]
        )

        agent = AgentService(retriever, llm, tool_registry)
        query = (
            "1. Driver Eligibility & Required Documents\n"
            "● Minimum Age: Renters must be at least 21 years old."
        )
        return await agent.answer(query)

    result = asyncio.run(scenario())

    assert result.answer == "Unfortunately, nothing is available for those dates."
