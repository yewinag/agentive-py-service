from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.tools.tool import ToolMetadata


class ToolCall(BaseModel):
    """The model's request to invoke one tool. `id` distinguishes this
    call within a response (a model may request more than one tool in a
    single turn) and is what the corresponding tool-result message must
    reference back via LLMMessage.tool_call_id.
    """

    id: str
    tool_name: str
    arguments: dict = Field(default_factory=dict)


class LLMMessage(BaseModel):
    """One turn in the conversation as an LLMProvider needs to see it -
    NOT the same type as ConversationMessage (app/conversation), which is
    the persisted domain record with its own concerns (created_at,
    user-facing content validation). This is the LLM-facing projection
    built fresh for each call: it can carry provider-agnostic tool-call/
    tool-result structure that ConversationMessage deliberately does not
    (see README's Agent section on why that stays out of persistent
    conversation storage).
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_call_id: Optional[str] = None  # set when role == "tool": which ToolCall this answers
    tool_calls: list[ToolCall] = Field(default_factory=list)  # set when role == "assistant" and tools were requested


class LLMRequest(BaseModel):
    """What an LLMProvider is asked to do. `tools` defaults to empty -
    AnswerGenerator never sets it (staying unaware tools exist at all);
    AgentService populates it from ToolRegistry.list_tools(), reusing
    ToolMetadata directly rather than duplicating the tool schema here.
    """

    messages: list[LLMMessage]
    tools: list[ToolMetadata] = Field(default_factory=list)


class LLMResponse(BaseModel):
    """An LLMProvider's result. Exactly one of `text`/`tool_calls` is
    meaningful for a given response - a provider implementation resolves
    this itself (see OpenAIProvider) so callers never have to guess
    which one to trust.
    """

    text: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
