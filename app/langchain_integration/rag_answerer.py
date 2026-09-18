"""Adapts LangChainRagGenerationService (generation.py) to this
project's existing RagAnswerer shape (app/chats/service.py:
`answer(question, history=None) -> GroundedAnswer`) - the same shape
AgentService already implements. This is the ONLY file that bridges
LangChain-specific types (RagAnswer, RagSource, and transitively
Document/ChatOpenAI/ChatPromptTemplate) to this project's own
app/rag/models types (GroundedAnswer, AnswerSource); nothing outside
app/langchain_integration ever needs to know LangChain is involved.
"""
from typing import Optional

from app.conversation.models import ConversationMessage
from app.core.config import Settings
from app.langchain_integration.generation import LangChainRagGenerationService, get_rag_generation_service
from app.rag.models import AnswerSource, GroundedAnswer


class LangChainRagAnswerer:
    """RagAnswerer implementation backed by LangChainRagGenerationService.

    `history` is accepted only to satisfy the shared RagAnswerer
    interface AgentService also implements - it is intentionally not
    forwarded. LangChainRagGenerationService is stateless per Phase
    2.7.4's scope (one retrieval call, one prompt, one chat model call;
    no tool calling, no conversation memory) - adding conversation
    awareness to the LangChain path is a future phase's decision, not
    something to smuggle in here while wiring the wiring.
    """

    def __init__(self, generation_service: LangChainRagGenerationService) -> None:
        self._generation_service = generation_service

    async def answer(
        self, question: str, history: Optional[list[ConversationMessage]] = None
    ) -> GroundedAnswer:
        rag_answer = await self._generation_service.agenerate(question)
        return GroundedAnswer(
            answer=rag_answer.answer,
            sources=[
                AnswerSource(document_title=source.document_title, section_heading=source.section_heading)
                for source in rag_answer.sources
            ],
        )


def get_langchain_rag_answerer(settings: Settings) -> LangChainRagAnswerer:
    """Composition point, matching every other module's plain-function
    pattern (get_rag_generation_service, get_langchain_retriever).
    """
    return LangChainRagAnswerer(generation_service=get_rag_generation_service(settings))
