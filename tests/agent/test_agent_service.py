import asyncio

import pytest
from pydantic import BaseModel

from app.agent.service import AgentService
from app.conversation.models import ConversationMessage
from app.knowledge.models import DocumentChunk
from app.knowledge.vector_store import VectorSearchResult
from app.llm.exceptions import LLMProviderError
from app.llm.fake_provider import FakeLLMProvider
from app.llm.models import LLMResponse, ToolCall
from app.rag.answer_generator import NOT_AVAILABLE_ANSWER
from app.tools.exceptions import BusinessServiceUnavailableError, ToolExecutionError, ToolInputError
from app.tools.registry import ToolRegistry
from app.tools.tool import ToolMetadata


class StubRetriever:
    def __init__(self, results=None):
        self.calls = []
        self._results = results if results is not None else []

    async def retrieve(self, query, top_k=None, min_score=None):
        self.calls.append(query)
        return self._results


class StubToolResult(BaseModel):
    value: str = "ok"


class StubTool:
    """Satisfies Tool structurally, unrelated to CheckVehicleAvailabilityTool -
    keeps AgentService's own tests independent of Step 14's concrete tool.
    """

    def __init__(self, name="stub_tool", result=None, error=None):
        self._name = name
        self._result = result if result is not None else StubToolResult()
        self._error = error
        self.received_inputs = []

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name=self._name, description="A stub tool.", input_schema={"type": "object"})

    async def execute(self, raw_input: dict):
        self.received_inputs.append(raw_input)
        if self._error:
            raise self._error
        return self._result


class RecordingLLMProvider:
    """Records every LLMRequest it receives, returning configured
    responses in order - used where a test needs to inspect exactly what
    was sent (e.g. that tools were offered, or omitted on the follow-up
    call), which FakeLLMProvider deliberately doesn't track itself.
    """

    def __init__(self, responses):
        self.requests = []
        self._responses = list(responses)

    async def generate(self, request):
        self.requests.append(request)
        return self._responses.pop(0)


def _result(document_title="Doc", section_heading="1. Section", text="text") -> VectorSearchResult:
    return VectorSearchResult(
        chunk=DocumentChunk(
            id=f"{document_title}-{section_heading}",
            document_id="doc-1",
            document_title=document_title,
            section_heading=section_heading,
            text=text,
            position=0,
        ),
        score=0.9,
    )


def _registry(*tools) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def test_answer_returns_not_available_when_retrieval_is_empty_without_calling_llm():
    class ExplodingLLMProvider:
        async def generate(self, request):
            raise AssertionError("LLM must not be called when retrieval is empty")

    agent = AgentService(StubRetriever(results=[]), ExplodingLLMProvider(), ToolRegistry())

    result = asyncio.run(agent.answer("something unrelated"))

    assert result.answer == NOT_AVAILABLE_ANSWER
    assert result.sources == []


def test_answer_returns_final_text_directly_when_no_tool_call_is_requested():
    llm = FakeLLMProvider(responses=[LLMResponse(text="Direct final answer.")])
    agent = AgentService(StubRetriever(results=[_result()]), llm, ToolRegistry())

    result = asyncio.run(agent.answer("What is the policy?"))

    assert result.answer == "Direct final answer."
    assert len(result.sources) == 1


def test_answer_offers_registered_tools_to_the_llm():
    llm = RecordingLLMProvider(responses=[LLMResponse(text="answer")])
    tool = StubTool(name="stub_tool")
    agent = AgentService(StubRetriever(results=[_result()]), llm, _registry(tool))

    asyncio.run(agent.answer("question"))

    assert [t.name for t in llm.requests[0].tools] == ["stub_tool"]


def test_answer_executes_a_requested_tool_and_returns_the_final_answer():
    tool = StubTool(name="stub_tool", result=StubToolResult(value="42 vehicles available"))
    tool_call = ToolCall(id="call-1", tool_name="stub_tool", arguments={"x": 1})
    llm = FakeLLMProvider(
        responses=[
            LLMResponse(tool_calls=[tool_call]),
            LLMResponse(text="Based on the tool result, here is your answer."),
        ]
    )
    agent = AgentService(StubRetriever(results=[_result()]), llm, _registry(tool))

    result = asyncio.run(agent.answer("Is a car available?"))

    assert result.answer == "Based on the tool result, here is your answer."
    assert tool.received_inputs == [{"x": 1}]


def test_tool_result_is_sent_back_to_the_llm_as_a_tool_message():
    tool = StubTool(name="stub_tool", result=StubToolResult(value="specific-result-marker"))
    tool_call = ToolCall(id="call-1", tool_name="stub_tool", arguments={})
    llm = RecordingLLMProvider(
        responses=[LLMResponse(tool_calls=[tool_call]), LLMResponse(text="final")]
    )
    agent = AgentService(StubRetriever(results=[_result()]), llm, _registry(tool))

    asyncio.run(agent.answer("question"))

    second_request = llm.requests[1]
    tool_messages = [m for m in second_request.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call-1"
    assert "specific-result-marker" in tool_messages[0].content


def test_second_call_offers_no_tools_enforcing_the_bounded_round():
    tool = StubTool(name="stub_tool")
    tool_call = ToolCall(id="call-1", tool_name="stub_tool", arguments={})
    llm = RecordingLLMProvider(
        responses=[LLMResponse(tool_calls=[tool_call]), LLMResponse(text="final")]
    )
    agent = AgentService(StubRetriever(results=[_result()]), llm, _registry(tool))

    asyncio.run(agent.answer("question"))

    assert llm.requests[1].tools == []


def test_unknown_tool_is_rejected_safely_and_llm_gets_a_final_chance():
    tool_call = ToolCall(id="call-1", tool_name="does_not_exist", arguments={})
    llm = FakeLLMProvider(
        responses=[
            LLMResponse(tool_calls=[tool_call]),
            LLMResponse(text="I could not find that tool."),
        ]
    )
    agent = AgentService(StubRetriever(results=[_result()]), llm, ToolRegistry())

    result = asyncio.run(agent.answer("question"))

    assert result.answer == "I could not find that tool."


def test_invalid_tool_arguments_are_rejected_safely():
    tool = StubTool(name="stub_tool", error=ToolInputError("bad input"))
    tool_call = ToolCall(id="call-1", tool_name="stub_tool", arguments={})
    llm = FakeLLMProvider(
        responses=[LLMResponse(tool_calls=[tool_call]), LLMResponse(text="Could not check that.")]
    )
    agent = AgentService(StubRetriever(results=[_result()]), llm, _registry(tool))

    result = asyncio.run(agent.answer("question"))

    assert result.answer == "Could not check that."


def test_business_service_failure_is_handled_gracefully():
    tool = StubTool(name="stub_tool", error=BusinessServiceUnavailableError("down"))
    tool_call = ToolCall(id="call-1", tool_name="stub_tool", arguments={})
    llm = FakeLLMProvider(
        responses=[LLMResponse(tool_calls=[tool_call]), LLMResponse(text="Try again later.")]
    )
    agent = AgentService(StubRetriever(results=[_result()]), llm, _registry(tool))

    result = asyncio.run(agent.answer("question"))

    assert result.answer == "Try again later."


def test_tool_execution_failure_is_handled_gracefully():
    tool = StubTool(name="stub_tool", error=ToolExecutionError("failed"))
    tool_call = ToolCall(id="call-1", tool_name="stub_tool", arguments={})
    llm = FakeLLMProvider(
        responses=[LLMResponse(tool_calls=[tool_call]), LLMResponse(text="Something went wrong.")]
    )
    agent = AgentService(StubRetriever(results=[_result()]), llm, _registry(tool))

    result = asyncio.run(agent.answer("question"))

    assert result.answer == "Something went wrong."


def test_llm_failure_on_first_call_propagates():
    class FailingLLMProvider:
        async def generate(self, request):
            raise LLMProviderError("boom")

    agent = AgentService(StubRetriever(results=[_result()]), FailingLLMProvider(), ToolRegistry())

    with pytest.raises(LLMProviderError):
        asyncio.run(agent.answer("question"))


def test_llm_failure_on_second_call_after_tool_execution_propagates():
    tool = StubTool(name="stub_tool")
    tool_call = ToolCall(id="call-1", tool_name="stub_tool", arguments={})

    class FlakyLLMProvider:
        def __init__(self):
            self._first_call = True

        async def generate(self, request):
            if self._first_call:
                self._first_call = False
                return LLMResponse(tool_calls=[tool_call])
            raise LLMProviderError("boom on second call")

    agent = AgentService(StubRetriever(results=[_result()]), FlakyLLMProvider(), _registry(tool))

    with pytest.raises(LLMProviderError):
        asyncio.run(agent.answer("question"))


def test_maximum_tool_call_round_is_enforced():
    """If a provider ignores tools=[] and still requests a tool on the
    follow-up call (a misbehaving provider), AgentService refuses to
    continue rather than looping - the bounded policy fails loudly
    instead of silently allowing a second round.
    """
    tool = StubTool(name="stub_tool")
    tool_call = ToolCall(id="call-1", tool_name="stub_tool", arguments={})
    llm = FakeLLMProvider(
        responses=[LLMResponse(tool_calls=[tool_call]), LLMResponse(tool_calls=[tool_call])]
    )
    agent = AgentService(StubRetriever(results=[_result()]), llm, _registry(tool))

    with pytest.raises(LLMProviderError):
        asyncio.run(agent.answer("question"))


def test_answer_passes_history_into_the_prompt():
    llm = FakeLLMProvider()  # default echo behavior
    agent = AgentService(StubRetriever(results=[_result()]), llm, ToolRegistry())
    history = [ConversationMessage(role="user", content="Earlier question")]

    result = asyncio.run(agent.answer("Follow-up question", history=history))

    assert "Conversation so far:" in result.answer
    assert "User: Earlier question" in result.answer


def test_retrieval_is_called_with_only_the_current_question():
    retriever = StubRetriever(results=[_result()])
    agent = AgentService(retriever, FakeLLMProvider(), ToolRegistry())
    history = [ConversationMessage(role="user", content="unrelated earlier question")]

    asyncio.run(agent.answer("current question", history=history))

    assert retriever.calls == ["current question"]
