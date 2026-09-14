from app.llm.models import LLMMessage, LLMRequest, LLMResponse, ToolCall
from app.tools.tool import ToolMetadata


def test_llm_response_represents_final_text():
    response = LLMResponse(text="Here is your answer.")

    assert response.text == "Here is your answer."
    assert response.tool_calls == []


def test_llm_response_represents_a_tool_call():
    tool_call = ToolCall(id="call-1", tool_name="check_vehicle_availability", arguments={"category": "SUV"})

    response = LLMResponse(tool_calls=[tool_call])

    assert response.text is None
    assert response.tool_calls == [tool_call]


def test_tool_call_carries_name_arguments_and_id():
    tool_call = ToolCall(
        id="call-1",
        tool_name="check_vehicle_availability",
        arguments={"pickup_at": "2026-01-01T10:00:00"},
    )

    assert tool_call.id == "call-1"
    assert tool_call.tool_name == "check_vehicle_availability"
    assert tool_call.arguments == {"pickup_at": "2026-01-01T10:00:00"}


def test_tool_call_arguments_default_to_empty_dict():
    tool_call = ToolCall(id="call-1", tool_name="some_tool")

    assert tool_call.arguments == {}


def test_llm_message_defaults_have_no_tool_call_fields():
    message = LLMMessage(role="user", content="Hello")

    assert message.tool_call_id is None
    assert message.tool_calls == []


def test_llm_message_can_represent_a_tool_result():
    message = LLMMessage(role="tool", content='{"ok": true}', tool_call_id="call-1")

    assert message.role == "tool"
    assert message.tool_call_id == "call-1"


def test_llm_message_can_represent_an_assistant_tool_call_request():
    tool_call = ToolCall(id="call-1", tool_name="check_vehicle_availability", arguments={})

    message = LLMMessage(role="assistant", content=None, tool_calls=[tool_call])

    assert message.tool_calls == [tool_call]


def test_llm_request_tools_default_to_empty():
    request = LLMRequest(messages=[LLMMessage(role="user", content="Hi")])

    assert request.tools == []


def test_llm_request_reuses_tool_metadata_directly():
    metadata = ToolMetadata(name="check_vehicle_availability", description="...", input_schema={})

    request = LLMRequest(messages=[LLMMessage(role="user", content="Hi")], tools=[metadata])

    assert request.tools == [metadata]
