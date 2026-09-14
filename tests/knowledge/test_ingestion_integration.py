"""Ingests the two real, committed car-rental PDFs through the real
PdfDocumentExtractor + SectionAwareChunker (not fakes), proving
IngestionService "supports processing the current knowledge documents
deterministically". Embedding stays fake so this needs no network/API
key - only extraction and chunking use real project content.
"""
import asyncio
from pathlib import Path

import pytest

from app.knowledge.chunking import SectionAwareChunker
from app.knowledge.fake_embedding_provider import FakeEmbeddingProvider
from app.knowledge.in_memory_vector_store import InMemoryVectorStore
from app.knowledge.ingestion import IngestionService
from app.knowledge.models import DocumentSource
from app.knowledge.pdf_extractor import PdfDocumentExtractor

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"

pytestmark = pytest.mark.skipif(
    not KNOWLEDGE_DIR.exists(),
    reason="data/knowledge/ not present in this checkout",
)


def test_ingest_processes_both_real_pdfs_deterministically():
    store = InMemoryVectorStore()
    service = IngestionService(
        extractor=PdfDocumentExtractor(),
        chunker=SectionAwareChunker(),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )
    sources = [
        DocumentSource(
            id="car-rental-services.pdf",
            title="Services Breakdown",
            content=str(KNOWLEDGE_DIR / "car-rental-services.pdf"),
        ),
        DocumentSource(
            id="car-rental-policies.pdf",
            title="Terms & Rental Policies",
            content=str(KNOWLEDGE_DIR / "car-rental-policies.pdf"),
        ),
    ]

    first_run_count = asyncio.run(service.ingest(sources))
    second_run_count = asyncio.run(service.ingest(sources))  # re-ingest: upsert, not duplication

    # Matches Step 7's known chunk counts for these exact PDFs: 4 (services) + 6 (policies).
    assert first_run_count == 10
    assert second_run_count == 10

    results = asyncio.run(store.search(query_embedding=[0.0] * 8, top_k=100))
    assert len(results) == 10  # re-ingesting didn't duplicate chunks
    document_ids = {result.chunk.document_id for result in results}
    assert document_ids == {"car-rental-services.pdf", "car-rental-policies.pdf"}
