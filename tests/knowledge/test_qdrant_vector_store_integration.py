"""Integration coverage against a real Qdrant instance.

Skipped unless QDRANT_URL is set - the rest of the suite (and CI by
default) never needs a running Qdrant. To run this file locally:

    docker compose up -d qdrant
    QDRANT_URL=http://localhost:6333 python -m pytest tests/knowledge/test_qdrant_vector_store_integration.py -v
"""
import asyncio
import os
import uuid
from contextlib import asynccontextmanager

import pytest

from app.knowledge.models import DocumentChunk
from app.knowledge.qdrant_vector_store import QdrantVectorStore
from app.knowledge.vector_store import VectorRecord

QDRANT_URL = os.environ.get("QDRANT_URL")

pytestmark = pytest.mark.skipif(
    not QDRANT_URL, reason="QDRANT_URL not set - skipping real Qdrant integration test"
)


def _chunk(chunk_id: str, text: str, position: int = 0, document_id: str = "doc-1") -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id=document_id,
        document_title="Integration Test Doc",
        section_heading="1. Section",
        text=text,
        position=position,
    )


@asynccontextmanager
async def _temporary_store(dimensions: int = 4):
    """Everything (collection creation, the test body, and teardown)
    must run on the same asyncio event loop, the same reasoning
    test_pgvector_store_integration.py documents for its own
    asyncpg-backed store - each test below wraps its entire scenario in
    one asyncio.run() call rather than splitting setup into a fixture.
    """
    collection_name = f"test_chunk_embeddings_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(url=QDRANT_URL, dimensions=dimensions, collection_name=collection_name)
    await store.ensure_collection()
    try:
        yield store
    finally:
        await store.delete_collection()
        await store.close()


def test_ensure_collection_is_idempotent():
    async def scenario():
        async with _temporary_store() as store:
            await store.ensure_collection()  # second call must not raise/recreate
            results = await store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=5)
            assert results == []

    asyncio.run(scenario())


def test_add_and_search_returns_most_similar_chunk_first():
    async def scenario():
        async with _temporary_store() as store:
            await store.add(
                [
                    VectorRecord(chunk=_chunk("c1", "cats are great pets"), embedding=[1.0, 0.0, 0.0, 0.0]),
                    VectorRecord(chunk=_chunk("c2", "dogs are loyal"), embedding=[0.0, 1.0, 0.0, 0.0]),
                ]
            )

            results = await store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=1)

            assert len(results) == 1
            assert results[0].chunk.id == "c1"
            assert results[0].score > 0.99

    asyncio.run(scenario())


def test_search_preserves_chunk_metadata():
    async def scenario():
        async with _temporary_store() as store:
            await store.add(
                [
                    VectorRecord(
                        chunk=_chunk("c1", "cancellation terms", position=2, document_id="04-cancellation-policy.pdf"),
                        embedding=[1.0, 0.0, 0.0, 0.0],
                    )
                ]
            )

            results = await store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=1)

            assert len(results) == 1
            chunk = results[0].chunk
            assert chunk.id == "c1"
            assert chunk.document_id == "04-cancellation-policy.pdf"
            assert chunk.document_title == "Integration Test Doc"
            assert chunk.section_heading == "1. Section"
            assert chunk.text == "cancellation terms"
            assert chunk.position == 2

    asyncio.run(scenario())


def test_search_on_empty_collection_returns_empty_list():
    async def scenario():
        async with _temporary_store() as store:
            results = await store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=5)
            assert results == []

    asyncio.run(scenario())


def test_search_against_a_never_created_collection_returns_empty_list_not_an_error():
    """search() must handle "collection doesn't exist yet" cleanly - a
    fresh QdrantVectorStore pointed at a collection nothing has ever
    ingested into yet, rather than only an ensured-but-empty one.
    """
    async def scenario():
        collection_name = f"test_never_created_{uuid.uuid4().hex[:8]}"
        store = QdrantVectorStore(url=QDRANT_URL, dimensions=4, collection_name=collection_name)
        try:
            results = await store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=5)
            assert results == []
        finally:
            await store.close()

    asyncio.run(scenario())


def test_add_upserts_by_chunk_id():
    async def scenario():
        async with _temporary_store() as store:
            await store.add([VectorRecord(chunk=_chunk("c1", "original"), embedding=[1.0, 0.0, 0.0, 0.0])])
            await store.add([VectorRecord(chunk=_chunk("c1", "updated"), embedding=[1.0, 0.0, 0.0, 0.0])])

            results = await store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=10)

            assert len(results) == 1
            assert results[0].chunk.text == "updated"

    asyncio.run(scenario())


def test_search_respects_top_k_across_multiple_documents():
    async def scenario():
        async with _temporary_store() as store:
            await store.add(
                [
                    VectorRecord(
                        chunk=_chunk(f"doc{d}-c{i}", f"text {d}-{i}", position=i, document_id=f"doc-{d}"),
                        embedding=[1.0, float(i), 0.0, 0.0],
                    )
                    for d in range(3)
                    for i in range(2)
                ]
            )

            results = await store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=2)

            assert len(results) == 2

    asyncio.run(scenario())


def test_persists_across_a_new_client_connecting_to_the_same_collection():
    """Proves persistence: a second QdrantVectorStore instance (a new
    client, simulating an application restart) connecting to the same
    named collection sees data the first instance wrote, because the
    data lives in Qdrant's own storage (the docker-compose volume), not
    in the Python process.
    """
    async def scenario():
        collection_name = f"test_restart_{uuid.uuid4().hex[:8]}"
        first = QdrantVectorStore(url=QDRANT_URL, dimensions=4, collection_name=collection_name)
        await first.ensure_collection()
        await first.add([VectorRecord(chunk=_chunk("c1", "persisted chunk"), embedding=[1.0, 0.0, 0.0, 0.0])])
        await first.close()

        second = QdrantVectorStore(url=QDRANT_URL, dimensions=4, collection_name=collection_name)
        try:
            results = await second.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=5)
            assert len(results) == 1
            assert results[0].chunk.text == "persisted chunk"
        finally:
            await second.delete_collection()
            await second.close()

    asyncio.run(scenario())
