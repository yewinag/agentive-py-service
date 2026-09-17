"""Integration coverage for the explicit Qdrant ingestion command
(app/knowledge/ingest.py) against a real Qdrant instance, using the
real, committed canonical PDFs and the real PdfDocumentExtractor +
SectionAwareChunker (FakeEmbeddingProvider stays in - see README's
"Keep fake embeddings for this phase" note; this step is about
persistence, not semantic quality).

Skipped unless QDRANT_URL is set - the rest of the suite (and CI by
default) never needs a running Qdrant. To run this file locally:

    docker compose up -d qdrant
    QDRANT_URL=http://localhost:6333 VECTOR_STORE_PROVIDER=qdrant \\
        python -m pytest tests/knowledge/test_qdrant_ingestion_command_integration.py -v
"""
import asyncio
import os
import uuid

import pytest

from app.core.config import Settings
from app.knowledge.fake_embedding_provider import FakeEmbeddingProvider
from app.knowledge.ingest import run
from app.knowledge.qdrant_vector_store import QdrantVectorStore

QDRANT_URL = os.environ.get("QDRANT_URL")

pytestmark = pytest.mark.skipif(
    not QDRANT_URL, reason="QDRANT_URL not set - skipping real Qdrant integration test"
)


def _settings(collection_name: str) -> Settings:
    return Settings(
        vector_store_provider="qdrant",
        qdrant_url=QDRANT_URL,
        qdrant_collection=collection_name,
        embedding_provider="fake",
    )


def test_ingestion_command_discovers_all_six_pdfs_and_produces_22_chunks(capsys):
    collection_name = f"test_ingest_cmd_{uuid.uuid4().hex[:8]}"
    settings = _settings(collection_name)

    async def scenario():
        exit_code = await run(settings, reset=False)
        output = capsys.readouterr().out
        store = QdrantVectorStore(url=QDRANT_URL, dimensions=8, collection_name=collection_name)
        try:
            assert exit_code == 0
            assert "Documents discovered: 6" in output
            assert "Chunks generated: 22" in output
            assert "Vectors upserted: 22" in output
            assert "Status: SUCCESS" in output

            results = await store.search(query_embedding=[0.0] * 8, top_k=100)
            assert len(results) == 22
        finally:
            await store.delete_collection()
            await store.close()

    asyncio.run(scenario())


def test_ingestion_command_is_idempotent_across_repeated_runs():
    collection_name = f"test_ingest_idempotent_{uuid.uuid4().hex[:8]}"
    settings = _settings(collection_name)

    async def scenario():
        store = QdrantVectorStore(url=QDRANT_URL, dimensions=8, collection_name=collection_name)
        try:
            first_exit = await run(settings, reset=False)
            second_exit = await run(settings, reset=False)

            assert first_exit == 0
            assert second_exit == 0

            results = await store.search(query_embedding=[0.0] * 8, top_k=100)
            assert len(results) == 22  # still 22, not 44 - upserted by chunk id, not duplicated
        finally:
            await store.delete_collection()
            await store.close()

    asyncio.run(scenario())


def test_vectors_persist_across_a_fresh_client_after_ingestion():
    """Simulates an application restart: a brand-new QdrantVectorStore
    (new client, new process in spirit) built from nothing but the
    collection name must see everything the ingestion command wrote.
    """
    collection_name = f"test_ingest_persist_{uuid.uuid4().hex[:8]}"
    settings = _settings(collection_name)

    async def scenario():
        exit_code = await run(settings, reset=False)
        assert exit_code == 0

        fresh_store = QdrantVectorStore(url=QDRANT_URL, dimensions=8, collection_name=collection_name)
        try:
            embedding_provider = FakeEmbeddingProvider()
            [query_vector] = await embedding_provider.embed(["Booking Policy"])
            results = await fresh_store.search(query_vector, top_k=1)

            assert len(results) == 1
            assert results[0].chunk.document_title == "Booking Policy"
        finally:
            await fresh_store.delete_collection()
            await fresh_store.close()

    asyncio.run(scenario())


def test_reset_flag_recreates_the_collection_without_duplicating():
    collection_name = f"test_ingest_reset_{uuid.uuid4().hex[:8]}"
    settings = _settings(collection_name)

    async def scenario():
        store = QdrantVectorStore(url=QDRANT_URL, dimensions=8, collection_name=collection_name)
        try:
            first_exit = await run(settings, reset=False)
            second_exit = await run(settings, reset=True)

            assert first_exit == 0
            assert second_exit == 0

            results = await store.search(query_embedding=[0.0] * 8, top_k=100)
            assert len(results) == 22
        finally:
            await store.delete_collection()
            await store.close()

    asyncio.run(scenario())
