import pytest

from app.core.config import Settings
from app.llm.dependencies import get_llm_provider
from app.llm.fake_provider import FakeLLMProvider
from app.llm.openai_provider import OpenAIProvider


def test_get_llm_provider_returns_fake_by_default():
    settings = Settings(llm_provider="fake")

    provider = get_llm_provider(settings)

    assert isinstance(provider, FakeLLMProvider)


def test_get_llm_provider_returns_openai_provider_when_configured():
    settings = Settings(
        llm_provider="openai", openai_api_key="sk-test", openai_model="gpt-4o-mini"
    )

    provider = get_llm_provider(settings)

    assert isinstance(provider, OpenAIProvider)


def test_get_llm_provider_fails_clearly_when_openai_key_missing():
    settings = Settings(llm_provider="openai", openai_api_key=None)

    with pytest.raises(RuntimeError):
        get_llm_provider(settings)
