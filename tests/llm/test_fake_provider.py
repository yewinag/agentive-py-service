import asyncio

from app.llm.fake_provider import FakeLLMProvider


def test_fake_provider_returns_deterministic_reply():
    provider = FakeLLMProvider()

    reply = asyncio.run(provider.generate_reply("Hello"))

    assert reply == "[fake-llm-reply] Hello"
