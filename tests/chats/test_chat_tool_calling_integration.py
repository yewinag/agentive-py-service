"""End-to-end proof that POST /api/v1/chat can produce a tool-assisted
answer, with no external infrastructure:

    FakeEmbeddingProvider -> InMemoryVectorStore -> Retriever
    -> FakeLLMProvider (scripted tool-call then final-text sequence)
    -> CheckVehicleAvailabilityTool -> FakeBusinessServiceClient
    -> ToolRegistry -> AgentService -> InMemoryConversationStore
    -> ChatService -> FastAPI /api/v1/chat

No real OpenAI call, no NestJS dependency, no PostgreSQL requirement.
The user-facing contract is unchanged: reply/sources/conversation_id -
nothing in the response reveals that a tool was involved.
"""
import asyncio
from datetime import datetime, timedelta

from app.agent.service import AgentService
from app.chats.service import ChatService, get_chat_service
from app.conversation.in_memory_store import InMemoryConversationStore
from app.knowledge.chunking import SectionAwareChunker
from app.knowledge.fake_embedding_provider import FakeEmbeddingProvider
from app.knowledge.fake_extractor import FakeDocumentExtractor
from app.knowledge.in_memory_vector_store import InMemoryVectorStore
from app.knowledge.ingestion import IngestionService
from app.knowledge.models import DocumentSource
from app.knowledge.retriever import VectorRetriever
from app.llm.fake_provider import FakeLLMProvider
from app.llm.models import LLMResponse, ToolCall
from app.main import app
from app.tools.business_client import VehicleAvailability
from app.tools.check_vehicle_availability import TOOL_NAME, CheckVehicleAvailabilityTool
from app.tools.fake_business_client import FakeBusinessServiceClient
from app.tools.registry import ToolRegistry

POLICY_TEXT = (
    "1. Driver Eligibility & Required Documents\n"
    "● Minimum Age: Renters must be at least 21 years old.\n"
)
NOW = datetime(2026, 6, 1, 10, 0)


async def _build_tool_assisted_chat_service() -> ChatService:
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
    tool_registry = ToolRegistry()
    tool_registry.register(CheckVehicleAvailabilityTool(FakeBusinessServiceClient(vehicles=[vehicle])))

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
            LLMResponse(text="Yes, a Honda CR-V is available for those dates."),
        ]
    )

    agent = AgentService(retriever, llm, tool_registry)
    return ChatService(agent, InMemoryConversationStore(), history_window=6)


def test_chat_endpoint_returns_a_tool_assisted_answer(client):
    service = asyncio.run(_build_tool_assisted_chat_service())
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        query = (
            "1. Driver Eligibility & Required Documents\n"
            "● Minimum Age: Renters must be at least 21 years old."
        )
        response = client.post("/api/v1/chat", json={"message": query})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Yes, a Honda CR-V is available for those dates."
    assert isinstance(body["conversation_id"], str) and body["conversation_id"]
    assert body["sources"] == [
        {
            "document_title": "Terms & Rental Policies",
            "section_heading": "1. Driver Eligibility & Required Documents",
        }
    ]
    # The public contract reveals nothing about the tool call - no
    # tool-call trace, no internal message list, just the final answer.
    assert set(body.keys()) == {"reply", "sources", "conversation_id"}
