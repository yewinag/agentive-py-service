import math

from app.knowledge.vector_store import VectorRecord, VectorSearchResult


class InMemoryVectorStore:
    """Deterministic, dependency-free VectorStore implementation.
    Satisfies VectorStore structurally - it never needs to inherit from
    it. The default (vector_store_provider=memory) so local development
    and the test suite never require a running database.

    Similarity is plain cosine similarity in pure Python - no numpy
    needed at this scale (a handful of chunks today), and it keeps this
    module dependency-free the same way SectionAwareChunker is.
    """

    def __init__(self) -> None:
        self._records: dict = {}

    async def add(self, records: list[VectorRecord]) -> None:
        for record in records:
            self._records[record.chunk.id] = record

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[VectorSearchResult]:
        results = [
            VectorSearchResult(
                chunk=record.chunk,
                score=_cosine_similarity(query_embedding, record.embedding),
            )
            for record in self._records.values()
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)
