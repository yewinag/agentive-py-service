import hashlib


class FakeEmbeddingProvider:
    """Deterministic stand-in for a real embedding call. Satisfies
    EmbeddingProvider structurally - it never needs to inherit from it.

    Each vector is derived from a hash of its input text rather than
    zeros/random: distinct inputs reliably produce distinct vectors,
    which matters for tests that check ordering (a chunk's embedding
    must correspond to its own text, not just be "a vector").
    """

    def __init__(self, dimensions: int = 8) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i] / 255 for i in range(self._dimensions)]
