import pytest

from app.conversation.in_memory_store import InMemoryConversationStore
from app.conversation.store import get_conversation_store, reset_default_conversation_store
from app.core.config import Settings


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_default_conversation_store()
    yield
    reset_default_conversation_store()


def test_get_conversation_store_returns_in_memory_by_default():
    settings = Settings(conversation_store_provider="memory")

    store = get_conversation_store(settings)

    assert isinstance(store, InMemoryConversationStore)


def test_get_conversation_store_returns_the_same_instance_across_calls():
    settings = Settings(conversation_store_provider="memory")

    first = get_conversation_store(settings)
    second = get_conversation_store(settings)

    assert first is second


def test_reset_default_conversation_store_clears_the_singleton():
    settings = Settings(conversation_store_provider="memory")
    first = get_conversation_store(settings)

    reset_default_conversation_store()
    second = get_conversation_store(settings)

    assert first is not second


def test_get_conversation_store_rejects_unknown_provider():
    settings = Settings(conversation_store_provider="not-a-real-store")

    with pytest.raises(NotImplementedError):
        get_conversation_store(settings)
