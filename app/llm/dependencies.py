from fastapi import Depends

from app.core.config import Settings, get_settings
from app.llm.fake_provider import FakeLLMProvider
from app.llm.openai_provider import OpenAIProvider
from app.llm.provider import LLMProvider


def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    """Composition point for LLMProvider. This is the only place that
    decides which concrete provider is used - a new provider gets added
    here as another branch, and nothing else in the app changes.
    """
    if settings.llm_provider == "fake":
        return FakeLLMProvider()

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
            )
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )

    raise NotImplementedError(
        f"LLM provider '{settings.llm_provider}' is not implemented yet"
    )
