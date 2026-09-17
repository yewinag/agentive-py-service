import asyncio

import pytest

from app.core.config import Settings
from app.knowledge.bootstrap import bootstrap_default_knowledge_base, discover_canonical_sources
from app.knowledge.vector_store import get_vector_store, reset_default_vector_store

CANONICAL_FILENAMES = {
    "01-rental-services.pdf",
    "02-rental-policies.pdf",
    "03-booking-policy.pdf",
    "04-cancellation-policy.pdf",
    "05-payment-policy.pdf",
    "06-pickup-return-policy.pdf",
}


def test_discover_canonical_sources_finds_all_six_pdfs_with_clean_titles():
    sources = discover_canonical_sources()

    assert {source.id for source in sources} == CANONICAL_FILENAMES
    assert all(source.title and not source.title.endswith(".pdf") for source in sources)
    # every source's content is a real path to a PDF that exists on disk
    assert all(source.content.endswith(".pdf") for source in sources)


@pytest.fixture(autouse=True)
def _reset_in_memory_singleton():
    reset_default_vector_store()
    yield
    reset_default_vector_store()


def test_bootstrap_ingests_the_real_pdfs_into_the_default_store():
    settings = Settings(vector_store_provider="memory")

    count = asyncio.run(bootstrap_default_knowledge_base(settings))

    assert count == 22  # 4+4+4+3+4+3 across the six canonical PDFs


def test_bootstrap_populates_the_same_store_get_vector_store_returns():
    settings = Settings(vector_store_provider="memory")

    asyncio.run(bootstrap_default_knowledge_base(settings))

    store = get_vector_store(settings)
    results = asyncio.run(store.search(query_embedding=[0.0] * 8, top_k=100))
    assert len(results) == 22


def test_bootstrap_is_idempotent_across_repeated_calls():
    settings = Settings(vector_store_provider="memory")

    first_count = asyncio.run(bootstrap_default_knowledge_base(settings))
    second_count = asyncio.run(bootstrap_default_knowledge_base(settings))

    store = get_vector_store(settings)
    results = asyncio.run(store.search(query_embedding=[0.0] * 8, top_k=100))
    assert first_count == 22
    assert second_count == 22
    assert len(results) == 22  # re-running never duplicates chunks


def test_bootstrap_is_a_no_op_for_pgvector():
    settings = Settings(
        vector_store_provider="pgvector",
        database_url="postgresql+asyncpg://user:pass@localhost/db",
    )

    count = asyncio.run(bootstrap_default_knowledge_base(settings))

    assert count == 0


def test_bootstrap_is_a_no_op_for_qdrant():
    """Qdrant is a persistent, already-shared store - just like pgvector,
    it must never be auto-ingested into on every app boot. See
    app/knowledge/ingest.py for Qdrant's real, explicit ingestion path.
    """
    settings = Settings(vector_store_provider="qdrant")

    count = asyncio.run(bootstrap_default_knowledge_base(settings))

    assert count == 0
