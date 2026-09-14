from typing import Protocol

from app.llm.models import LLMRequest, LLMResponse


class LLMProvider(Protocol):
    """The boundary ChatService/AnswerGenerator/AgentService code
    against: WHAT the application needs from an LLM. Concrete providers
    (fake today, a real SDK-backed one later) implement this
    structurally - no explicit subclassing needed.

    Evolved in Step 15 from a single generate_reply(message: str) -> str
    method into this richer generate(request) -> response contract, so a
    caller can offer tools and receive back either final text or a
    request to call one - without any provider-specific type (OpenAI's
    SDK types included) ever appearing outside app/llm/openai_provider.py.
    generate_reply was retired rather than kept alongside this: one
    unambiguous contract beats two overlapping ones, and every existing
    caller (AnswerGenerator) needed only a trivial, behavior-preserving
    update (build an LLMRequest with a single user message, read
    `.text` back) - see README's Agent section for the full reasoning.
    """

    async def generate(self, request: LLMRequest) -> LLMResponse: ...
