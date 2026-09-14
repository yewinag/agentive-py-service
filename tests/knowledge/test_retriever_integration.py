"""End-to-end retrieval using the real FakeEmbeddingProvider and
InMemoryVectorStore (no mocks, no external infrastructure). Demonstrates
query -> embedding provider -> vector store -> relevant chunk, the full
Step 10 flow, deterministically: FakeEmbeddingProvider's vectors are
hash-derived, so embedding the exact same text as a stored chunk always
yields the top, near-1.0-score match for that chunk.
"""
import asyncio

from app.knowledge.fake_embedding_provider import FakeEmbeddingProvider
from app.knowledge.in_memory_vector_store import InMemoryVectorStore
from app.knowledge.models import DocumentChunk
from app.knowledge.retriever import VectorRetriever
from app.knowledge.vector_store import VectorRecord

_CHUNK_TEXTS = {
    "eligibility": "Minimum Age: Renters must be at least 21 years old.",
    "payment": "Accepted Payment Methods: Credit Card, Bank Transfer, or PromptPay.",
    "cancellation": "Free Cancellation: Up to 48 hours prior to the scheduled pickup time.",
}


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id="policies.pdf",
        document_title="Terms & Rental Policies",
        section_heading="1. Driver Eligibility & Required Documents",
        text=text,
        position=0,
    )


def test_retrieve_returns_the_chunk_matching_the_query_text():
    async def scenario():
        embedding_provider = FakeEmbeddingProvider()
        vector_store = InMemoryVectorStore()
        retriever = VectorRetriever(embedding_provider, vector_store, default_top_k=5)

        records = []
        for chunk_id, text in _CHUNK_TEXTS.items():
            [embedding] = await embedding_provider.embed([text])
            records.append(VectorRecord(chunk=_chunk(chunk_id, text), embedding=embedding))
        await vector_store.add(records)

        results = await retriever.retrieve(_CHUNK_TEXTS["eligibility"], top_k=1)

        assert len(results) == 1
        assert results[0].chunk.id == "eligibility"
        assert results[0].score > 0.99

    asyncio.run(scenario())


def test_retrieve_top_k_limits_results_across_the_whole_pipeline():
    async def scenario():
        embedding_provider = FakeEmbeddingProvider()
        vector_store = InMemoryVectorStore()
        retriever = VectorRetriever(embedding_provider, vector_store, default_top_k=5)

        records = []
        for chunk_id, text in _CHUNK_TEXTS.items():
            [embedding] = await embedding_provider.embed([text])
            records.append(VectorRecord(chunk=_chunk(chunk_id, text), embedding=embedding))
        await vector_store.add(records)

        results = await retriever.retrieve(_CHUNK_TEXTS["payment"], top_k=2)

        assert len(results) == 2
        assert results[0].chunk.id == "payment"
        assert results[0].score >= results[1].score

    asyncio.run(scenario())
