from typing import Optional

from app.conversation.models import ConversationMessage
from app.core.config import Settings
from app.knowledge.retriever import Retriever
from app.llm.exceptions import LLMProviderError
from app.llm.models import LLMMessage, LLMRequest, ToolCall
from app.llm.provider import LLMProvider
from app.rag.answer_generator import (
    NOT_AVAILABLE_ANSWER,
    build_context,
    build_prompt,
    to_source,
)
from app.rag.models import GroundedAnswer
from app.tools.exceptions import BusinessServiceUnavailableError, ToolExecutionError, ToolInputError
from app.tools.registry import ToolRegistry

MAX_TOOL_ROUNDS = 1  # scope: exactly one tool-call round permitted this step - see README


class AgentService:
    """Orchestrates retrieval + a bounded tool-calling round + grounded
    answer generation. A strict superset of AnswerGenerator's capability
    - reuses its context/prompt-building helpers (build_context,
    build_prompt, to_source, NOT_AVAILABLE_ANSWER) rather than
    duplicating them, but needs its own top-level method: tool-calling
    requires inspecting the LLM's response *before* deciding whether to
    return it or execute a tool and continue, which AnswerGenerator's
    single, non-branching call cannot express. AnswerGenerator itself is
    unchanged and remains a fully working, directly usable pure-RAG (no
    tool awareness) building block - see README's Agent section for the
    full reasoning behind not modifying or removing it.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
    ) -> None:
        self._retriever = retriever
        self._llm_provider = llm_provider
        self._tool_registry = tool_registry

    async def answer(
        self, question: str, history: Optional[list[ConversationMessage]] = None
    ) -> GroundedAnswer:
        # Retrieval stays focused on the current question alone, exactly
        # like AnswerGenerator - see README's RAG interaction note.
        results = await self._retriever.retrieve(question)

        if not results:
            return GroundedAnswer(answer=NOT_AVAILABLE_ANSWER, sources=[])

        prompt = build_prompt(question, build_context(results), history)
        messages = [LLMMessage(role="user", content=prompt)]
        available_tools = self._tool_registry.list_tools()

        response = await self._llm_provider.generate(
            LLMRequest(messages=messages, tools=available_tools)
        )

        if response.tool_calls:
            messages.append(
                LLMMessage(role="assistant", content=response.text, tool_calls=response.tool_calls)
            )
            for tool_call in response.tool_calls:
                messages.append(await self._execute_tool_call(tool_call))

            # Bounded policy (MAX_TOOL_ROUNDS = 1): the follow-up call
            # offers no tools at all, so the model cannot request
            # another - structurally guaranteed, not just checked after
            # the fact. Offering fewer tools than were available on a
            # later round is exactly the mechanism that makes "one
            # round" a hard limit rather than a convention.
            response = await self._llm_provider.generate(LLMRequest(messages=messages, tools=[]))

            if response.tool_calls:
                raise LLMProviderError(
                    "Model requested another tool call beyond the allowed round; refusing to continue."
                )

        return GroundedAnswer(
            answer=response.text,
            sources=[to_source(result) for result in results],
        )

    async def _execute_tool_call(self, tool_call: ToolCall) -> LLMMessage:
        """Resolves and executes a requested tool, translating any
        failure into a short, generic error string fed back to the model
        as the tool's result - never a raw exception message, and never
        anything but a registered tool's own execute(). An unknown or
        failing tool does not abort the request: the model gets a
        chance to explain it couldn't complete the request, the same
        way a real assistant would.
        """
        tool = self._tool_registry.resolve(tool_call.tool_name)
        if tool is None:
            content = f"Error: no tool named '{tool_call.tool_name}' is available."
        else:
            try:
                result = await tool.execute(tool_call.arguments)
                content = result.model_dump_json()
            except ToolInputError:
                content = "Error: the arguments provided for this tool were invalid."
            except BusinessServiceUnavailableError:
                content = "Error: the business service is currently unavailable."
            except ToolExecutionError:
                content = "Error: the tool failed to complete."

        return LLMMessage(role="tool", content=content, tool_call_id=tool_call.id)


def get_agent_service(settings: Settings) -> AgentService:
    """Composition point - a plain function, not FastAPI Depends()-wired
    itself; ChatService's get_chat_service() is the FastAPI-facing bridge
    that calls this, the same pattern as get_answer_generator()."""
    from app.knowledge.retriever import get_retriever
    from app.llm.dependencies import get_llm_provider
    from app.tools.registry import get_tool_registry

    return AgentService(
        retriever=get_retriever(settings),
        llm_provider=get_llm_provider(settings),
        tool_registry=get_tool_registry(settings),
    )
