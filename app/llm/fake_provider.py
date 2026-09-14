class FakeLLMProvider:
    """Deterministic stand-in for a real LLM call. Satisfies LLMProvider
    structurally - it never needs to inherit from it.
    """

    async def generate_reply(self, message: str) -> str:
        return f"[fake-llm-reply] {message}"
