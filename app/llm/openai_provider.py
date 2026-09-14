from typing import Optional

from openai import AsyncOpenAI, OpenAIError

from app.llm.exceptions import LLMProviderError


class OpenAIProvider:
    """LLMProvider implementation backed by the OpenAI API. The openai SDK
    is an implementation detail of this module alone - nothing outside
    app/llm ever imports it.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self._model = model
        self._client = client or AsyncOpenAI(api_key=api_key)

    async def generate_reply(self, message: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": message}],
            )
        except OpenAIError as exc:
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc

        content = response.choices[0].message.content
        if content is None:
            raise LLMProviderError("OpenAI response contained no message content")
        return content
