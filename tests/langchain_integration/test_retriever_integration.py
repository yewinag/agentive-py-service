"""Integration coverage for KnowledgeBaseRetriever
(app/langchain_integration/retriever.py) against the REAL, already
populated Qdrant collection (knowledge_chunk_embeddings) and REAL
OpenAI embeddings - no fakes, no simulated results, per this phase's
explicit "do not use fake/simulated retrieval results" instruction.

Skipped unless both QDRANT_URL and OPENAI_API_KEY are set as real
environment variables (not read from .env - see conftest.py's
_isolate_settings_from_the_real_env_file, which deliberately keeps
tests hermetic against a developer's local .env). This never runs in
the default `pytest -v` invocation and makes real, billed OpenAI calls
when it does. To run it locally:

    docker compose up -d qdrant
    QDRANT_URL=http://localhost:6333 OPENAI_API_KEY=sk-... \\
        python -m pytest tests/langchain_integration/test_retriever_integration.py -v

Does not ingest, reset, or otherwise modify the collection - read-only
against whatever Phase 2.7.2 already persisted there.
"""
import asyncio
import os

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client import AsyncQdrantClient

from app.core.config import Settings
from app.langchain_integration.retriever import KnowledgeBaseRetriever, get_langchain_retriever

QDRANT_URL = os.environ.get("QDRANT_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COLLECTION_NAME = "knowledge_chunk_embeddings"

pytestmark = pytest.mark.skipif(
    not (QDRANT_URL and OPENAI_API_KEY),
    reason="QDRANT_URL and/or OPENAI_API_KEY not set - skipping real Qdrant+OpenAI integration test",
)

EXPECTED_SOURCES = {
    "What is the minimum age to rent a car?": "02-rental-policies.pdf",
    "What happens if I cancel less than 24 hours before pickup?": "04-cancellation-policy.pdf",
    "How much is the security deposit?": "05-payment-policy.pdf",
}

METADATA_FIELDS = {"chunk_id", "document_id", "document_title", "section_heading", "position", "score"}


def _settings() -> Settings:
    return Settings(
        vector_store_provider="qdrant",
        embedding_provider="openai",
        qdrant_url=QDRANT_URL,
        qdrant_collection=COLLECTION_NAME,
        openai_api_key=OPENAI_API_KEY,
    )


async def _existing_collection_names() -> set:
    client = AsyncQdrantClient(url=QDRANT_URL)
    try:
        response = await client.get_collections()
        return {c.name for c in response.collections}
    finally:
        await client.close()


def test_get_langchain_retriever_returns_a_base_retriever():
    retriever = get_langchain_retriever(_settings())

    assert isinstance(retriever, KnowledgeBaseRetriever)
    assert isinstance(retriever, BaseRetriever)


def test_does_not_create_a_second_qdrant_collection():
    async def scenario():
        before = await _existing_collection_names()
        assert COLLECTION_NAME in before

        retriever = get_langchain_retriever(_settings())
        await retriever.ainvoke("What is the minimum age to rent a car?")

        after = await _existing_collection_names()
        assert after == before  # exactly the same set - nothing added, nothing removed

    asyncio.run(scenario())


@pytest.mark.parametrize("question, expected_document_id", list(EXPECTED_SOURCES.items()))
def test_retrieval_returns_langchain_documents_with_expected_source_in_top_results(
    question, expected_document_id
):
    async def scenario():
        retriever = get_langchain_retriever(_settings(), top_k=5)

        documents = await retriever.ainvoke(question)

        assert len(documents) > 0
        for document in documents:
            assert isinstance(document, Document)
            assert isinstance(document.page_content, str) and document.page_content.strip()
            assert METADATA_FIELDS.issubset(document.metadata.keys())

        retrieved_document_ids = {d.metadata["document_id"] for d in documents}
        assert expected_document_id in retrieved_document_ids, (
            f"Expected {expected_document_id!r} among top results for {question!r}, "
            f"got {retrieved_document_ids!r}"
        )

    asyncio.run(scenario())


def test_metadata_preserves_every_existing_payload_field_for_the_top_result():
    async def scenario():
        retriever = get_langchain_retriever(_settings(), top_k=1)

        [document] = await retriever.ainvoke("How much is the security deposit?")

        assert document.metadata["document_id"] == "05-payment-policy.pdf"
        assert document.metadata["document_title"] == "Payment Policy"
        assert document.metadata["section_heading"] == "2. Security Deposit"
        assert document.metadata["chunk_id"].startswith("05-payment-policy.pdf-chunk-")
        assert isinstance(document.metadata["position"], int)
        assert isinstance(document.metadata["score"], float)
        assert document.page_content.startswith("2. Security Deposit")

    asyncio.run(scenario())
