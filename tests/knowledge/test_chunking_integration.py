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

# (filename, title, expected section headings) for each of the six canonical PDFs.
CANONICAL_DOCUMENTS = [
    (
        "01-rental-services.pdf",
        "Rental Services",
        {"1. Overview of Services", "2. Rental Options", "3. Vehicle Categories & Inclusions"},
    ),
    (
        "02-rental-policies.pdf",
        "Rental Policies",
        {"1. Eligibility Requirements", "2. Insurance & Protection", "3. Vehicle Usage Restrictions"},
    ),
    (
        "03-booking-policy.pdf",
        "Booking Policy",
        {"1. Reservation Process", "2. Required Information for Booking", "3. Modifications & Extensions"},
    ),
    (
        "04-cancellation-policy.pdf",
        "Cancellation Policy",
        {"1. Cancellation Terms", "2. Refund Processing"},
    ),
    (
        "05-payment-policy.pdf",
        "Payment Policy",
        {"1. Accepted Payment Methods", "2. Security Deposit", "3. Additional Charges & Fines"},
    ),
    (
        "06-pickup-return-policy.pdf",
        "Pickup and Return Policy",
        {"1. Pickup Procedure", "2. Vehicle Return Procedure"},
    ),
]


def _chunk(filename: str, title: str):
    source = DocumentSource(id=filename, title=title, content=str(KNOWLEDGE_DIR / filename))
    extracted = asyncio.run(PdfDocumentExtractor().extract(source))
    return SectionAwareChunker().chunk(extracted)


def _report(filename: str, chunks: list) -> None:
    sizes = [len(c.text) for c in chunks]
    print(f"\n{filename}: {len(chunks)} chunks, "
          f"min={min(sizes)} max={max(sizes)} avg={sum(sizes) / len(sizes):.0f} chars")


@pytest.mark.parametrize("filename, title, expected_headings", CANONICAL_DOCUMENTS)
def test_canonical_pdf_chunks_one_chunk_per_section_with_headings_preserved(filename, title, expected_headings):
    chunks = _chunk(filename, title)
    _report(filename, chunks)

    headed_chunks = [c for c in chunks if c.section_heading is not None]
    assert {c.section_heading for c in headed_chunks} == expected_headings
    for chunk in headed_chunks:
        assert chunk.text.startswith(chunk.section_heading)

    positions = [c.position for c in chunks]
    assert positions == list(range(len(chunks)))

    # 1 preamble chunk + N section chunks, one section chunk per numbered heading
    assert len(chunks) == 1 + len(expected_headings)


def test_no_canonical_section_needs_further_splitting_at_default_size():
    """Documents this small produce exactly one chunk per section - the
    long-section fallback path exists for future, larger documents but
    isn't exercised by today's six PDFs.
    """
    total_chunks = 0
    for filename, title, expected_headings in CANONICAL_DOCUMENTS:
        chunks = _chunk(filename, title)
        assert len(chunks) == 1 + len(expected_headings)
        total_chunks += len(chunks)

    # 4 + 4 + 4 + 3 + 4 + 3, per this step's measured chunk counts.
    assert total_chunks == 22
