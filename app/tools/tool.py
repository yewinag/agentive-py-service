from typing import Protocol

from pydantic import BaseModel


class ToolMetadata(BaseModel):
    """Application-level tool description a future LLM/function-calling
    layer can be given - deliberately not shaped like any one provider's
    function-calling format (e.g. OpenAI's `{"type": "function", ...}`
    wrapper). `input_schema` is a plain JSON Schema dict, generated from
    the tool's own Pydantic input model (`InputModel.model_json_schema()`)
    - JSON Schema is what every major function-calling API already
    expects its parameters shaped like, so no provider-specific
    adaptation is needed to hand this to one, but nothing here commits
    to a specific provider's envelope around it either.
    """

    name: str
    description: str
    input_schema: dict


class Tool(Protocol):
    """The boundary a future agent/tool-calling layer codes against: WHAT
    it needs to describe a capability and invoke it. Concrete tools
    (CheckVehicleAvailabilityTool today) implement this structurally -
    the same pattern as every other Protocol in this codebase.

    `execute` takes raw, unvalidated input (a plain dict, the shape an
    LLM's function-call arguments would actually arrive in) and returns
    the tool's own result model - there is no generic ToolResult
    wrapper, since each tool's result is genuinely different data and a
    wrapper would only erase that. Failures are signaled via the
    app/tools/exceptions.py hierarchy, not a success/error result field.
    """

    @property
    def metadata(self) -> ToolMetadata: ...

    async def execute(self, raw_input: dict) -> BaseModel: ...
