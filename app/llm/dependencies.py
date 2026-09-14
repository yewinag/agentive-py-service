from fastapi import Depends

from app.core.config import Settings, get_settings
from app.llm.fake_provider import FakeLLMProvider
from app.llm.provider import LLMProvider


def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    """Composition point for LLMProvider. This is the only place that
    decides which concrete provider is used - a real one gets added here
    as another branch once it exists, and nothing else in the app changes.
    """
    if settings.llm_provider != "fake":
        raise NotImplementedError(
            f"LLM provider '{settings.llm_provider}' is not implemented yet"
        )
    return FakeLLMProvider()
