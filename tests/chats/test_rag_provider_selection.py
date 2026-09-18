import pytest

from app.agent.service import AgentService
from app.chats.service import get_rag_answerer
from app.core.config import Settings
from app.langchain_integration.rag_answerer import LangChainRagAnswerer


def test_get_rag_answerer_returns_agent_service_by_default():
    settings = Settings()

    answerer = get_rag_answerer(settings)

    assert isinstance(answerer, AgentService)


def test_get_rag_answerer_returns_agent_service_for_existing_provider():
    settings = Settings(rag_provider="existing")

    answerer = get_rag_answerer(settings)

    assert isinstance(answerer, AgentService)


def test_get_rag_answerer_returns_langchain_rag_answerer_when_selected():
    settings = Settings(
        rag_provider="langchain",
        embedding_provider="openai",
        openai_api_key="sk-test",
        vector_store_provider="qdrant",
    )

    answerer = get_rag_answerer(settings)

    assert isinstance(answerer, LangChainRagAnswerer)


def test_get_rag_answerer_rejects_unknown_provider():
    settings = Settings(rag_provider="not-a-real-provider")

    with pytest.raises(NotImplementedError):
        get_rag_answerer(settings)
