import asyncio

from app.knowledge.in_memory_vector_store import InMemoryVectorStore
from app.knowledge.models import DocumentChunk
from app.knowledge.vector_store import VectorRecord


def _chunk(chunk_id: str, text: str = "text", position: int = 0) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id="doc-1",
        document_title="Sample Doc",
        section_heading="1. Section",
        text=text,
        position=position,
    )


def test_search_on_empty_store_returns_empty_list():
    store = InMemoryVectorStore()

    results = asyncio.run(store.search(query_embedding=[1.0, 0.0, 0.0], top_k=5))

    assert results == []


def test_search_ranks_by_similarity_descending():
    store = InMemoryVectorStore()
    asyncio.run(
        store.add(
            [
                VectorRecord(chunk=_chunk("close"), embedding=[0.9, 0.1, 0.0]),
                VectorRecord(chunk=_chunk("exact"), embedding=[1.0, 0.0, 0.0]),
                VectorRecord(chunk=_chunk("orthogonal"), embedding=[0.0, 1.0, 0.0]),
            ]
        )
    )

    results = asyncio.run(store.search(query_embedding=[1.0, 0.0, 0.0], top_k=3))

    assert [r.chunk.id for r in results] == ["exact", "close", "orthogonal"]
    assert results[0].score > results[1].score > results[2].score
    assert results[0].score == 1.0


def test_search_respects_top_k():
    store = InMemoryVectorStore()
    asyncio.run(
        store.add(
            [
                VectorRecord(chunk=_chunk(f"c{i}"), embedding=[1.0, float(i), 0.0])
                for i in range(5)
            ]
        )
    )

    results = asyncio.run(store.search(query_embedding=[1.0, 0.0, 0.0], top_k=2))

    assert len(results) == 2


def test_search_preserves_chunk_metadata():
    store = InMemoryVectorStore()
    chunk = DocumentChunk(
        id="c1",
        document_id="policies.pdf",
        document_title="Terms & Rental Policies",
        section_heading="1. Driver Eligibility & Required Documents",
        text="Minimum Age: Renters must be at least 21 years old.",
        position=1,
    )
    asyncio.run(store.add([VectorRecord(chunk=chunk, embedding=[1.0, 0.0])]))

    results = asyncio.run(store.search(query_embedding=[1.0, 0.0], top_k=1))

    result_chunk = results[0].chunk
    assert result_chunk.document_id == "policies.pdf"
    assert result_chunk.document_title == "Terms & Rental Policies"
    assert result_chunk.section_heading == "1. Driver Eligibility & Required Documents"
    assert result_chunk.text == "Minimum Age: Renters must be at least 21 years old."


def test_add_upserts_by_chunk_id():
    store = InMemoryVectorStore()
    asyncio.run(store.add([VectorRecord(chunk=_chunk("c1", text="original"), embedding=[1.0, 0.0])]))
    asyncio.run(store.add([VectorRecord(chunk=_chunk("c1", text="updated"), embedding=[1.0, 0.0])]))

    results = asyncio.run(store.search(query_embedding=[1.0, 0.0], top_k=10))

    assert len(results) == 1
    assert results[0].chunk.text == "updated"
