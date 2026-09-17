"""Integration coverage against the real, committed knowledge-base PDFs
(data/knowledge/). Unlike test_pdf_extractor.py, this intentionally
exercises the actual production source documents so a future pdfplumber
upgrade or a bad PDF replacement is caught here, not first in production.
"""
import asyncio
from pathlib import Path

import pytest

from app.knowledge.models import DocumentSource
from app.knowledge.pdf_extractor import PdfDocumentExtractor

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"

pytestmark = pytest.mark.skipif(
    not KNOWLEDGE_DIR.exists(),
    reason="data/knowledge/ not present in this checkout",
)


def _extract(filename: str, title: str):
    source = DocumentSource(id=filename, title=title, content=str(KNOWLEDGE_DIR / filename))
    return asyncio.run(PdfDocumentExtractor().extract(source))


@pytest.mark.parametrize(
    "filename, title, expected_snippets",
    [
        ("01-rental-services.pdf", "Rental Services", ["Chauffeur Services", "Suvarnabhumi Airport"]),
        ("02-rental-policies.pdf", "Rental Policies", ["Minimum Age", "Collision Damage Waiver"]),
        ("03-booking-policy.pdf", "Booking Policy", ["Reservation ID", "Flight details"]),
        ("04-cancellation-policy.pdf", "Cancellation Policy", ["No-Show", "50% of the"]),
        ("05-payment-policy.pdf", "Payment Policy", ["Security Deposit", "PromptPay"]),
        ("06-pickup-return-policy.pdf", "Pickup and Return Policy", ["Vehicle Condition Form", "Grace Period"]),
    ],
)
def test_canonical_pdf_extracts_expected_content(filename, title, expected_snippets):
    extracted = _extract(filename, title)

    assert extracted.page_count == 1
    for snippet in expected_snippets:
        assert snippet in extracted.text
