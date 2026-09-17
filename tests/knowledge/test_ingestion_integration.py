"""Ingests the six real, committed canonical PDFs through the real
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

CANONICAL_TITLES = {
    "01-rental-services.pdf": "Rental Services",
    "02-rental-policies.pdf": "Rental Policies",
    "03-booking-policy.pdf": "Booking Policy",
    "04-cancellation-policy.pdf": "Cancellation Policy",
    "05-payment-policy.pdf": "Payment Policy",
    "06-pickup-return-policy.pdf": "Pickup and Return Policy",
}


def test_ingest_processes_all_six_canonical_pdfs_deterministically():
    store = InMemoryVectorStore()
    service = IngestionService(
        extractor=PdfDocumentExtractor(),
        chunker=SectionAwareChunker(),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )
    sources = [
        DocumentSource(id=filename, title=title, content=str(KNOWLEDGE_DIR / filename))
        for filename, title in CANONICAL_TITLES.items()
    ]

    first_run_count = asyncio.run(service.ingest(sources))
    second_run_count = asyncio.run(service.ingest(sources))  # re-ingest: upsert, not duplication

    # Matches the six canonical PDFs' measured chunk counts: 4+4+4+3+4+3 = 22.
    assert first_run_count == 22
    assert second_run_count == 22

    results = asyncio.run(store.search(query_embedding=[0.0] * 8, top_k=100))
    assert len(results) == 22  # re-ingesting didn't duplicate chunks
    document_ids = {result.chunk.document_id for result in results}
    assert document_ids == set(CANONICAL_TITLES)
