class ChatService:
    """Chat application logic. Currently a stub — real agent orchestration
    (LLM call, RAG, tool use) lands here in a later step without the
    router or schemas needing to change.
    """

    async def get_reply(self, message: str) -> str:
        return f"You said: {message}"


def get_chat_service() -> ChatService:
    return ChatService()
