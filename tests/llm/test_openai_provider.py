import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai import OpenAIError

from app.llm.exceptions import LLMProviderError
from app.llm.models import LLMMessage, LLMRequest, ToolCall
from app.llm.openai_provider import OpenAIProvider
from app.tools.tool import ToolMetadata


def _make_client(message):
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        )
    )


def _text_message(content):
    return SimpleNamespace(content=content, tool_calls=None)


def _tool_call_message(*tool_calls):
    return SimpleNamespace(content=None, tool_calls=list(tool_calls))


def _sdk_tool_call(call_id, name, arguments_json):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments_json)
    )


def test_generate_returns_text_response_from_message_content():
    client = _make_client(_text_message("Hello from OpenAI"))
    provider = OpenAIProvider(api_key="unused", model="gpt-4o-mini", client=client)

    response = asyncio.run(
        provider.generate(LLMRequest(messages=[LLMMessage(role="user", content="Hi")]))
    )

    assert response.text == "Hello from OpenAI"
    assert response.tool_calls == []
    client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
    )


def test_generate_omits_tools_kwarg_when_no_tools_offered():
    client = _make_client(_text_message("reply"))
    provider = OpenAIProvider(api_key="unused", model="gpt-4o-mini", client=client)

    asyncio.run(provider.generate(LLMRequest(messages=[LLMMessage(role="user", content="Hi")])))

    _, kwargs = client.chat.completions.create.call_args
    assert "tools" not in kwargs


def test_generate_converts_tool_metadata_into_openai_tool_definitions():
    client = _make_client(_text_message("reply"))
    provider = OpenAIProvider(api_key="unused", model="gpt-4o-mini", client=client)
    tool = ToolMetadata(
        name="check_vehicle_availability",
        description="Checks availability.",
        input_schema={"type": "object", "properties": {}, "required": []},
    )

    asyncio.run(
        provider.generate(
            LLMRequest(messages=[LLMMessage(role="user", content="Hi")], tools=[tool])
        )
    )

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "check_vehicle_availability",
                "description": "Checks availability.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]


def test_generate_parses_tool_calls_from_the_response():
    client = _make_client(
        _tool_call_message(
            _sdk_tool_call("call-1", "check_vehicle_availability", '{"category": "SUV"}')
        )
    )
    provider = OpenAIProvider(api_key="unused", model="gpt-4o-mini", client=client)

    response = asyncio.run(
        provider.generate(LLMRequest(messages=[LLMMessage(role="user", content="Hi")]))
    )

    assert response.text is None
    assert response.tool_calls == [
        ToolCall(id="call-1", tool_name="check_vehicle_availability", arguments={"category": "SUV"})
    ]


def test_generate_translates_assistant_tool_call_message_for_the_request():
    client = _make_client(_text_message("final answer"))
    provider = OpenAIProvider(api_key="unused", model="gpt-4o-mini", client=client)
    messages = [
        LLMMessage(role="user", content="Is a Camry free?"),
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call-1", tool_name="check_vehicle_availability", arguments={"category": "SUV"})],
        ),
        LLMMessage(role="tool", content='{"available_vehicles": []}', tool_call_id="call-1"),
    ]

    asyncio.run(provider.generate(LLMRequest(messages=messages)))

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["messages"] == [
        {"role": "user", "content": "Is a Camry free?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "check_vehicle_availability",
                        "arguments": json.dumps({"category": "SUV"}),
                    },
                }
            ],
        },
        {"role": "tool", "content": '{"available_vehicles": []}', "tool_call_id": "call-1"},
    ]


def test_generate_raises_for_malformed_tool_call_arguments():
    client = _make_client(
        _tool_call_message(_sdk_tool_call("call-1", "check_vehicle_availability", "not-json"))
    )
    provider = OpenAIProvider(api_key="unused", model="gpt-4o-mini", client=client)

    with pytest.raises(LLMProviderError):
        asyncio.run(provider.generate(LLMRequest(messages=[LLMMessage(role="user", content="Hi")])))


def test_generate_translates_sdk_errors_into_llm_provider_error():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=OpenAIError("boom")))
        )
    )
    provider = OpenAIProvider(api_key="unused", model="gpt-4o-mini", client=client)

    with pytest.raises(LLMProviderError):
        asyncio.run(provider.generate(LLMRequest(messages=[LLMMessage(role="user", content="Hi")])))


def test_generate_raises_when_response_has_no_content_and_no_tool_calls():
    client = _make_client(_text_message(None))
    provider = OpenAIProvider(api_key="unused", model="gpt-4o-mini", client=client)

    with pytest.raises(LLMProviderError):
        asyncio.run(provider.generate(LLMRequest(messages=[LLMMessage(role="user", content="Hi")])))
