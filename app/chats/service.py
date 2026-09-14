from fastapi import Depends

from app.core.config import Settings, get_settings
from app.rag.answer_generator import AnswerGenerator, get_answer_generator
from app.rag.models import GroundedAnswer


class ChatService:
    """Chat application logic. Delegates grounded-answer generation to
    AnswerGenerator - it knows WHAT it needs (an answer to a message),
    never HOW retrieval, embeddings, vector search, or the LLM SDK work.
    """

    def __init__(self, answer_generator: AnswerGenerator) -> None:
        self._answer_generator = answer_generator

    async def get_reply(self, message: str) -> GroundedAnswer:
        return await self._answer_generator.answer(message)


def get_chat_service(settings: Settings = Depends(get_settings)) -> ChatService:
    """FastAPI is now a real consumer of the RAG composition chain, so
    this is the one place that bridges from Depends()-resolved Settings
    to get_answer_generator() - the plain, framework-independent
    composition function AnswerGenerator/Retriever/VectorStore/
    EmbeddingProvider/LLMProvider are all still built through.
    """
    return ChatService(get_answer_generator(settings))
