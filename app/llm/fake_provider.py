from typing import Optional

from app.llm.exceptions import LLMProviderError
from app.llm.models import LLMRequest, LLMResponse


class FakeLLMProvider:
    """Deterministic stand-in for a real LLM call. Satisfies LLMProvider
    structurally - it never needs to inherit from it.

    With no configured responses, behaves exactly as before Step 15: a
    single-call echo of the request's most recent user message, prefixed
    "[fake-llm-reply] ". Pass `responses` to script an exact sequence
    (e.g. a tool-call response followed by a final-text response) for
    deterministically testing multi-call flows like AgentService's
    bounded tool round - never "smart" behavior, just an explicit,
    ordered list a test controls completely.
    """

    def __init__(self, responses: Optional[list[LLMResponse]] = None) -> None:
        self._responses = list(responses) if responses is not None else None

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if self._responses is not None:
            if not self._responses:
                raise LLMProviderError("FakeLLMProvider has no more configured responses")
            return self._responses.pop(0)

        last_user_message = next(
            (message.content for message in reversed(request.messages) if message.role == "user"),
            "",
        )
        return LLMResponse(text=f"[fake-llm-reply] {last_user_message}")
