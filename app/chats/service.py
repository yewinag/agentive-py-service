from typing import Optional, Protocol

from fastapi import Depends
from pydantic import BaseModel

from app.conversation.models import Conversation, ConversationMessage
from app.conversation.store import ConversationStore, get_conversation_store
from app.core.config import Settings, get_settings
from app.rag.models import GroundedAnswer


class RagAnswerer(Protocol):
    """The boundary ChatService codes against: WHAT it needs to turn a
    question (plus recent conversation history) into a GroundedAnswer -
    never HOW retrieval, tool-calling, or generation work. The same
    pattern as every other Protocol in this codebase (LLMProvider,
    EmbeddingProvider, VectorStore, Retriever).

    AgentService (app/agent/service.py, the default - retrieval +
    bounded tool-calling + LLMProvider) already implements this exactly,
    unchanged. LangChainRagAnswerer (app/langchain_integration/
    rag_answerer.py, selected via RAG_PROVIDER=langchain) is the other
    implementation - see get_rag_answerer() below.
    """

    async def answer(
        self, question: str, history: Optional[list[ConversationMessage]] = None
    ) -> GroundedAnswer: ...


class ChatResult(BaseModel):
    """ChatService.get_reply()'s return shape: the grounded answer plus
    which conversation it belongs to, so the router can echo
    conversation_id back to the client.
    """

    conversation_id: str
    answer: GroundedAnswer


class ChatService:
    """Chat application logic. Coordinates a ConversationStore (which
    conversation, its recent history) and a RagAnswerer (question +
    history -> GroundedAnswer) for one request - it knows WHAT it needs
    from each, never HOW retrieval, embeddings, vector search, tool
    execution, the LLM SDK, LangChain, or conversation persistence work.
    Depends on the RagAnswerer Protocol above, not concretely on
    AgentService - see get_rag_answerer() for how RAG_PROVIDER selects
    which implementation actually answers.
    """

    def __init__(
        self,
        rag_answerer: RagAnswerer,
        conversation_store: ConversationStore,
        history_window: int,
    ) -> None:
        self._rag_answerer = rag_answerer
        self._conversation_store = conversation_store
        self._history_window = history_window

    async def get_reply(
        self, message: str, conversation_id: Optional[str] = None
    ) -> ChatResult:
        conversation = await self._resolve_conversation(conversation_id)
        history = await self._conversation_store.get_recent_messages(
            conversation.id, limit=self._history_window
        )

        answer = await self._rag_answerer.answer(message, history=history)

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
    """FastAPI is now a real consumer of the RAG-answerer and
    conversation composition chains, so this is the one place that
    bridges from Depends()-resolved Settings to get_rag_answerer()/
    get_conversation_store() - the plain, framework-independent
    composition functions everything else is still built through.
    """
    return ChatService(
        rag_answerer=get_rag_answerer(settings),
        conversation_store=get_conversation_store(settings),
        history_window=settings.conversation_history_window,
    )


def get_rag_answerer(settings: Settings) -> RagAnswerer:
    """Selects which RagAnswerer implementation backs ChatService, from
    Settings.rag_provider. "existing" (default) returns AgentService
    itself - no adapter needed, it already implements this exact
    interface unchanged. "langchain" builds LangChainRagAnswerer, which
    wraps LangChainRagGenerationService (app/langchain_integration/) -
    imported here, deferred, only for that branch, the same pattern
    get_vector_store() already uses to keep optional backends out of
    this module's top-level imports. Either way, ChatService itself
    never imports AgentService or anything LangChain-specific - it only
    ever sees the RagAnswerer Protocol.
    """
    from app.agent.service import get_agent_service

    if settings.rag_provider == "existing":
        return get_agent_service(settings)

    if settings.rag_provider == "langchain":
        from app.langchain_integration.rag_answerer import get_langchain_rag_answerer

        return get_langchain_rag_answerer(settings)

    raise NotImplementedError(f"RAG provider '{settings.rag_provider}' is not implemented yet")
