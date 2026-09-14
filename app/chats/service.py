from typing import Optional

from fastapi import Depends
from pydantic import BaseModel

from app.agent.service import AgentService, get_agent_service
from app.conversation.models import Conversation, ConversationMessage
from app.conversation.store import ConversationStore, get_conversation_store
from app.core.config import Settings, get_settings
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
    conversation, its recent history) and an AgentService (retrieval +
    bounded tool-calling + grounded generation) for one request - it
    knows WHAT it needs from each, never HOW retrieval, embeddings,
    vector search, tool execution, the LLM SDK, or conversation
    persistence work. As of Step 15, delegates to AgentService rather
    than AnswerGenerator directly - AgentService is a strict superset of
    AnswerGenerator's capability (see app/agent/service.py), and its
    result type (GroundedAnswer) is identical, so this is the only line
    that changed here.
    """

    def __init__(
        self,
        agent_service: AgentService,
        conversation_store: ConversationStore,
        history_window: int,
    ) -> None:
        self._agent_service = agent_service
        self._conversation_store = conversation_store
        self._history_window = history_window

    async def get_reply(
        self, message: str, conversation_id: Optional[str] = None
    ) -> ChatResult:
        conversation = await self._resolve_conversation(conversation_id)
        history = await self._conversation_store.get_recent_messages(
            conversation.id, limit=self._history_window
        )

        answer = await self._agent_service.answer(message, history=history)

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
    """FastAPI is now a real consumer of the agent, RAG, and conversation
    composition chains, so this is the one place that bridges from
    Depends()-resolved Settings to get_agent_service()/
    get_conversation_store() - the plain, framework-independent
    composition functions everything else is still built through.
    """
    return ChatService(
        agent_service=get_agent_service(settings),
        conversation_store=get_conversation_store(settings),
        history_window=settings.conversation_history_window,
    )
