import json
from typing import Optional

from openai import AsyncOpenAI, OpenAIError

from app.llm.exceptions import LLMProviderError
from app.llm.models import LLMMessage, LLMRequest, LLMResponse, ToolCall


class OpenAIProvider:
    """LLMProvider implementation backed by the OpenAI API. The openai SDK
    is an implementation detail of this module alone - nothing outside
    app/llm ever imports it, including its tool-calling types: all
    translation between the application's provider-agnostic
    LLMRequest/LLMResponse/ToolCall and OpenAI's specific request/
    response shapes happens in this module's private helpers below.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self._model = model
        self._client = client or AsyncOpenAI(api_key=api_key)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        kwargs = {"model": self._model, "messages": _to_openai_messages(request.messages)}
        if request.tools:
            kwargs["tools"] = _to_openai_tools(request.tools)

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc

        message = response.choices[0].message

        if message.tool_calls:
            try:
                tool_calls = [_from_openai_tool_call(tool_call) for tool_call in message.tool_calls]
            except (json.JSONDecodeError, TypeError) as exc:
                raise LLMProviderError(
                    f"OpenAI returned malformed tool call arguments: {exc}"
                ) from exc
            return LLMResponse(tool_calls=tool_calls)

        if message.content is None:
            raise LLMProviderError("OpenAI response contained no message content")
        return LLMResponse(text=message.content)


def _to_openai_messages(messages: list[LLMMessage]) -> list[dict]:
    openai_messages = []
    for message in messages:
        if message.role == "tool":
            openai_messages.append(
                {"role": "tool", "content": message.content, "tool_call_id": message.tool_call_id}
            )
        elif message.role == "assistant" and message.tool_calls:
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.tool_name,
                                "arguments": json.dumps(tool_call.arguments),
                            },
                        }
                        for tool_call in message.tool_calls
                    ],
                }
            )
        else:
            openai_messages.append({"role": message.role, "content": message.content})
    return openai_messages


def _to_openai_tools(tools) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def _from_openai_tool_call(tool_call) -> ToolCall:
    return ToolCall(
        id=tool_call.id,
        tool_name=tool_call.function.name,
        arguments=json.loads(tool_call.function.arguments),
    )
