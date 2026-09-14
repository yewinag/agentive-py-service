import asyncio

import pytest

from app.llm.exceptions import LLMProviderError
from app.llm.fake_provider import FakeLLMProvider
from app.llm.models import LLMMessage, LLMRequest, LLMResponse, ToolCall


def test_default_behavior_echoes_the_most_recent_user_message():
    provider = FakeLLMProvider()

    response = asyncio.run(
        provider.generate(LLMRequest(messages=[LLMMessage(role="user", content="Hello")]))
    )

    assert response.text == "[fake-llm-reply] Hello"
    assert response.tool_calls == []


def test_default_behavior_uses_the_last_user_message_when_history_is_present():
    provider = FakeLLMProvider()
    messages = [
        LLMMessage(role="user", content="first"),
        LLMMessage(role="assistant", content="reply"),
        LLMMessage(role="user", content="second"),
    ]

    response = asyncio.run(provider.generate(LLMRequest(messages=messages)))

    assert response.text == "[fake-llm-reply] second"


def test_configured_sequence_returns_responses_in_order():
    tool_call_response = LLMResponse(
        tool_calls=[ToolCall(id="call-1", tool_name="check_vehicle_availability", arguments={})]
    )
    final_response = LLMResponse(text="Here is your answer.")
    provider = FakeLLMProvider(responses=[tool_call_response, final_response])
    request = LLMRequest(messages=[LLMMessage(role="user", content="question")])

    first = asyncio.run(provider.generate(request))
    second = asyncio.run(provider.generate(request))

    assert first is tool_call_response
    assert second is final_response


def test_configured_sequence_raises_once_exhausted():
    provider = FakeLLMProvider(responses=[LLMResponse(text="only response")])
    request = LLMRequest(messages=[LLMMessage(role="user", content="question")])

    asyncio.run(provider.generate(request))

    with pytest.raises(LLMProviderError):
        asyncio.run(provider.generate(request))
