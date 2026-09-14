from typing import Optional

from fastapi import Depends
from pydantic import BaseModel

from app.conversation.models import Conversation, ConversationMessage
from app.conversation.store import ConversationStore, get_conversation_store
from app.core.config import Settings, get_settings
from app.rag.answer_generator import AnswerGenerator, get_answer_generator
from app.rag.models import GroundedAnswer


class ChatResult(BaseModel):
    """ChatService.get_reply()'s return shape: the grounded answer plus
    which conversation it belongs to, so the router can echo
    conversation_id back to the client.
    """

    conversation_id: str
    answer: GroundedAnswer


class ChatService:
    """Chat application logic. Coordinates a ConversationStore (which
    conversation, its recent history) and an AnswerGenerator (retrieval +
    grounded generation) for one request - it knows WHAT it needs from
    each, never HOW retrieval, embeddings, vector search, the LLM SDK, or
    conversation persistence work.
    """

    def __init__(
        self,
        answer_generator: AnswerGenerator,
        conversation_store: ConversationStore,
        history_window: int,
    ) -> None:
        self._answer_generator = answer_generator
        self._conversation_store = conversation_store
        self._history_window = history_window

    async def get_reply(
        self, message: str, conversation_id: Optional[str] = None
    ) -> ChatResult:
        conversation = await self._resolve_conversation(conversation_id)
        history = await self._conversation_store.get_recent_messages(
            conversation.id, limit=self._history_window
        )

        answer = await self._answer_generator.answer(message, history=history)

        await self._conversation_store.append_message(
            conversation.id, ConversationMessage(role="user", content=message)
        )
        await self._conversation_store.append_message(
            conversation.id, ConversationMessage(role="assistant", content=answer.answer)
        )

        return ChatResult(conversation_id=conversation.id, answer=answer)

    async def _resolve_conversation(self, conversation_id: Optional[str]) -> Conversation:
        if conversation_id is not None:
            existing = await self._conversation_store.get(conversation_id)
            if existing is not None:
                return existing
            # An unknown id (e.g. the in-memory store restarted, or a
            # stale/mistyped client value) starts a fresh conversation
            # rather than erroring - the response's conversation_id tells
            # the client which id to use going forward.
        return await self._conversation_store.create()


def get_chat_service(settings: Settings = Depends(get_settings)) -> ChatService:
    """FastAPI is now a real consumer of both the RAG and conversation
    composition chains, so this is the one place that bridges from
    Depends()-resolved Settings to get_answer_generator()/
    get_conversation_store() - the plain, framework-independent
    composition functions everything else is still built through.
    """
    return ChatService(
        answer_generator=get_answer_generator(settings),
        conversation_store=get_conversation_store(settings),
        history_window=settings.conversation_history_window,
    )
