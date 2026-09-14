import pytest

from app.chats.service import get_chat_service
from app.knowledge.exceptions import EmbeddingProviderError, VectorStoreError
from app.llm.exceptions import LLMProviderError
from app.main import app


class FailingChatService:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def get_reply(self, message: str, conversation_id=None) -> None:
        raise self._error


def _override_with_failure(error: Exception):
    app.dependency_overrides[get_chat_service] = lambda: FailingChatService(error)


@pytest.mark.parametrize(
    "error",
    [
        EmbeddingProviderError("OpenAI embeddings request failed: connection reset"),
        VectorStoreError("could not connect to server: Connection refused"),
        LLMProviderError("OpenAI request failed: invalid_api_key"),
    ],
    ids=["embedding_failure", "vector_store_failure", "llm_failure"],
)
def test_dependency_failures_return_a_clean_503_without_leaking_details(client, error):
    _override_with_failure(error)
    try:
        response = client.post("/api/v1/chat", json={"message": "What is the deposit?"})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 503
    body = response.json()
    assert body == {"detail": "The chat service is temporarily unavailable. Please try again."}
    # the raw exception message (which could name a provider, a
    # connection string, or an API-key hint) must never reach the client
    assert str(error) not in response.text
