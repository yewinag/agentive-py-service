from fastapi import Depends

from app.llm.dependencies import get_llm_provider
from app.llm.provider import LLMProvider


class ChatService:
    """Chat application logic. Knows WHAT it needs from an LLM
    (a reply to a message) - never HOW to talk to a specific provider.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    async def get_reply(self, message: str) -> str:
        return await self._llm_provider.generate_reply(message)


def get_chat_service(
    llm_provider: LLMProvider = Depends(get_llm_provider),
) -> ChatService:
    return ChatService(llm_provider)
