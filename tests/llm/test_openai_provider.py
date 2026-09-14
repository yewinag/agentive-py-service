import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai import OpenAIError

from app.llm.exceptions import LLMProviderError
from app.llm.openai_provider import OpenAIProvider


def _make_client(content):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        )
    )


def test_generate_reply_returns_message_content_from_response():
    client = _make_client("Hello from OpenAI")
    provider = OpenAIProvider(api_key="unused", model="gpt-4o-mini", client=client)

    reply = asyncio.run(provider.generate_reply("Hi"))

    assert reply == "Hello from OpenAI"
    client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
    )


def test_generate_reply_translates_sdk_errors_into_llm_provider_error():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=OpenAIError("boom")))
        )
    )
    provider = OpenAIProvider(api_key="unused", model="gpt-4o-mini", client=client)

    with pytest.raises(LLMProviderError):
        asyncio.run(provider.generate_reply("Hi"))


def test_generate_reply_raises_when_response_has_no_content():
    client = _make_client(None)
    provider = OpenAIProvider(api_key="unused", model="gpt-4o-mini", client=client)

    with pytest.raises(LLMProviderError):
        asyncio.run(provider.generate_reply("Hi"))
