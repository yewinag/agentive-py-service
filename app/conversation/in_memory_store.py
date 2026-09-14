from typing import Optional

from app.conversation.exceptions import ConversationNotFoundError
from app.conversation.models import Conversation, ConversationMessage


class InMemoryConversationStore:
    """Deterministic, dependency-free ConversationStore implementation.
    Satisfies ConversationStore structurally - it never needs to inherit
    from it. The default (conversation_store_provider=memory): with no
    authentication or multi-process deployment yet, an in-process store
    is an honest match for what this stage needs, not a shortcut around
    a "real" implementation still to come.
    """

    def __init__(self) -> None:
        self._conversations: dict = {}

    async def create(self) -> Conversation:
        conversation = Conversation()
        self._conversations[conversation.id] = conversation
        return conversation

    async def get(self, conversation_id: str) -> Optional[Conversation]:
        return self._conversations.get(conversation_id)

    async def append_message(self, conversation_id: str, message: ConversationMessage) -> None:
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        conversation.messages.append(message)

    async def get_recent_messages(self, conversation_id: str, limit: int) -> list[ConversationMessage]:
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        if limit <= 0:
            return []
        return conversation.messages[-limit:]
