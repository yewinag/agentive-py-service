from typing import Optional

from openai import AsyncOpenAI, OpenAIError

from app.knowledge.exceptions import EmbeddingProviderError

# https://platform.openai.com/docs/guides/embeddings/embedding-models
# Output dimension is a fixed property of each OpenAI embedding model
# (the model's default; text-embedding-3-* can optionally be truncated
# via the API's `dimensions` parameter, which this provider does not
# use). Looked up here rather than hard-coded per instance so a caller
# never has to guess - and so an unrecognized model fails clearly
# instead of silently returning vectors of the wrong length.
_KNOWN_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider:
    """EmbeddingProvider implementation backed by the OpenAI API. The
    openai SDK is an implementation detail of this module alone -
    nothing outside app/knowledge ever imports it.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: Optional[int] = None,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self._model = model
        self._client = client or AsyncOpenAI(api_key=api_key)

        resolved_dimensions = dimensions or _KNOWN_MODEL_DIMENSIONS.get(model)
        if resolved_dimensions is None:
            raise EmbeddingProviderError(
                f"Unknown output dimensions for embedding model '{model}'. "
                "Pass dimensions=... explicitly to OpenAIEmbeddingProvider."
            )
        self._dimensions = resolved_dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            response = await self._client.embeddings.create(model=self._model, input=texts)
        except OpenAIError as exc:
            raise EmbeddingProviderError(f"OpenAI embedding request failed: {exc}") from exc

        # The API returns one Embedding per input with its own `.index`;
        # placing by index (not by response order) is what actually
        # guarantees the "same order as input" contract, rather than
        # assuming the SDK never reorders results.
        ordered: list = [None] * len(texts)
        for item in response.data:
            ordered[item.index] = item.embedding

        return ordered
