from typing import Optional, Protocol

from app.core.config import Settings
from app.conversation.models import Conversation, ConversationMessage


class ConversationStore(Protocol):
    """The boundary the chat layer codes against: WHAT it needs to
    create, look up, and append to conversations. Concrete stores
    (in-memory today, a PostgreSQL-backed one possibly later) implement
    this structurally - the same pattern as VectorStore/EmbeddingProvider.

    Deliberately not generic CRUD: no update/delete of individual
    messages, no listing all conversations - nothing in the chat flow
    needs those. `append_message` and `get_recent_messages` both assume
    the conversation exists (raising ConversationNotFoundError if not),
    since by the time either is called the caller has already
    established that via create() or a checked get().
    """

    async def create(self) -> Conversation:
        """Creates and stores a new, empty conversation with a fresh id."""
        ...

    async def get(self, conversation_id: str) -> Optional[Conversation]:
        """Returns the conversation, or None if conversation_id is unknown."""
        ...

    async def append_message(self, conversation_id: str, message: ConversationMessage) -> None:
        """Appends message to the conversation's history, preserving order."""
        ...

    async def get_recent_messages(self, conversation_id: str, limit: int) -> list[ConversationMessage]:
        """Returns up to the last `limit` messages, oldest first - ready
        to read top-to-bottom into a prompt.
        """
        ...


_shared_in_memory_conversation_store: Optional["InMemoryConversationStore"] = None  # noqa: F821


def get_conversation_store(settings: Settings) -> ConversationStore:
    """Composition point - a plain function, not FastAPI Depends()-wired,
    matching every other app/knowledge-style composition function
    (ChatService's get_chat_service() is still the one FastAPI-facing
    bridge). Only "memory" exists today; the setting still selects
    between provider names (matching vector_store_provider's shape) so a
    future PostgreSQL-backed store slots in without touching ChatService.

    Returns a process-wide singleton for "memory", the same reasoning as
    get_vector_store(): a fresh InMemoryConversationStore per call would
    mean one request's appended messages are invisible to the next.
    """
    from app.conversation.in_memory_store import InMemoryConversationStore

    if settings.conversation_store_provider == "memory":
        global _shared_in_memory_conversation_store
        if _shared_in_memory_conversation_store is None:
            _shared_in_memory_conversation_store = InMemoryConversationStore()
        return _shared_in_memory_conversation_store

    raise NotImplementedError(
        f"Conversation store '{settings.conversation_store_provider}' is not implemented yet"
    )


def reset_default_conversation_store() -> None:
    """Clears the process-wide InMemoryConversationStore singleton. For
    tests/dev isolation only - real request handling never needs this.
    """
    global _shared_in_memory_conversation_store
    _shared_in_memory_conversation_store = None
