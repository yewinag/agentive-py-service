"""Integration coverage against the real, committed knowledge-base PDFs
(data/knowledge/), chained through the actual extractor. Complements
test_pdf_extractor_integration.py by verifying the extractor's output is
still chunkable end-to-end and reports statistics useful for judging
whether the chunking strategy fits the real documents.
"""
import asyncio
from pathlib import Path

import pytest

from app.knowledge.chunking import SectionAwareChunker
from app.knowledge.models import DocumentSource
from app.knowledge.pdf_extractor import PdfDocumentExtractor

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"

pytestmark = pytest.mark.skipif(
    not KNOWLEDGE_DIR.exists(),
    reason="data/knowledge/ not present in this checkout",
)


def _chunk(filename: str, title: str):
    source = DocumentSource(id=filename, title=title, content=str(KNOWLEDGE_DIR / filename))
    extracted = asyncio.run(PdfDocumentExtractor().extract(source))
    return SectionAwareChunker().chunk(extracted)


def _report(filename: str, chunks: list) -> None:
    sizes = [len(c.text) for c in chunks]
    print(f"\n{filename}: {len(chunks)} chunks, "
          f"min={min(sizes)} max={max(sizes)} avg={sum(sizes) / len(sizes):.0f} chars")


def test_services_pdf_chunks_one_chunk_per_section_with_headings_preserved():
    chunks = _chunk("car-rental-services.pdf", "Services Breakdown")
    _report("car-rental-services.pdf", chunks)

    headed_chunks = [c for c in chunks if c.section_heading is not None]
    assert {c.section_heading for c in headed_chunks} == {
        "1. Daily & Short-Term Rentals",
        "2. Long-Term Rentals",
        "3. Special Mobility Services",
    }
    for chunk in headed_chunks:
        assert chunk.text.startswith(chunk.section_heading)

    positions = [c.position for c in chunks]
    assert positions == list(range(len(chunks)))


def test_policies_pdf_chunks_one_chunk_per_section_with_headings_preserved():
    chunks = _chunk("car-rental-policies.pdf", "Terms & Rental Policies")
    _report("car-rental-policies.pdf", chunks)

    headed_chunks = [c for c in chunks if c.section_heading is not None]
    assert {c.section_heading for c in headed_chunks} == {
        "1. Driver Eligibility & Required Documents",
        "2. Payment & Security Deposit",
        "3. Insurance & Damage Policy",
        "4. Cancellation & Modification Policy",
        "5. Vehicle Use Rules & Conditions",
    }
    for chunk in headed_chunks:
        assert chunk.text.startswith(chunk.section_heading)

    positions = [c.position for c in chunks]
    assert positions == list(range(len(chunks)))


def test_no_real_section_needs_further_splitting_at_default_size():
    """Documents this small produce exactly one chunk per section - the
    long-section fallback path exists for future, larger documents but
    isn't exercised by today's two PDFs.
    """
    services_chunks = _chunk("car-rental-services.pdf", "Services Breakdown")
    policies_chunks = _chunk("car-rental-policies.pdf", "Terms & Rental Policies")

    # 1 preamble chunk + N section chunks, one section chunk per numbered heading
    assert len(services_chunks) == 1 + 3
    assert len(policies_chunks) == 1 + 5
