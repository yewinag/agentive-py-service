from typing import Protocol


class LLMProvider(Protocol):
    """The boundary ChatService codes against: WHAT the application needs
    from an LLM. Concrete providers (fake today, a real SDK-backed one
    later) implement this structurally - no explicit subclassing needed.
    """

    async def generate_reply(self, message: str) -> str: ...
