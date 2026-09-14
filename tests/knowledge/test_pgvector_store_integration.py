"""Integration coverage against a real PostgreSQL + pgvector instance.

Skipped unless DATABASE_URL is set - the rest of the suite (and CI by
default) never needs a running database. To run this file locally:

    createdb agentive_test
    psql agentive_test -c "CREATE EXTENSION IF NOT EXISTS vector;"
    DATABASE_URL=postgresql+asyncpg://localhost/agentive_test \
        python -m pytest tests/knowledge/test_pgvector_store_integration.py -v
"""
import asyncio
import os
import uuid
from contextlib import asynccontextmanager

import pytest

from app.knowledge.models import DocumentChunk
from app.knowledge.pgvector_store import PgVectorStore
from app.knowledge.vector_store import VectorRecord

DATABASE_URL = os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="DATABASE_URL not set - skipping real Postgres integration test"
)


def _chunk(chunk_id: str, text: str, position: int = 0) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id="doc-1",
        document_title="Integration Test Doc",
        section_heading="1. Section",
        text=text,
        position=position,
    )


@asynccontextmanager
async def _temporary_store(dimensions: int = 4):
    """Everything (engine creation, schema setup, the test body, and
    teardown) must run on the same asyncio event loop - asyncpg's
    connection pool binds to whichever loop first uses it. Each test
    below therefore wraps its entire scenario in one asyncio.run() call,
    rather than splitting setup into a separate synchronous fixture.
    """
    table_name = f"test_chunk_embeddings_{uuid.uuid4().hex[:8]}"
    store = PgVectorStore(database_url=DATABASE_URL, dimensions=dimensions, table_name=table_name)
    await store.create_schema()
    try:
        yield store
    finally:
        await store.drop_schema()
        await store.dispose()


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


def test_search_on_empty_table_returns_empty_list():
    async def scenario():
        async with _temporary_store() as store:
            results = await store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=5)
            assert results == []

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


def test_search_respects_top_k():
    async def scenario():
        async with _temporary_store() as store:
            await store.add(
                [
                    VectorRecord(chunk=_chunk(f"c{i}", f"text {i}"), embedding=[1.0, float(i), 0.0, 0.0])
                    for i in range(5)
                ]
            )

            results = await store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=2)

            assert len(results) == 2

    asyncio.run(scenario())
