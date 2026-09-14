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


def test_services_pdf_extracts_expected_content():
    extracted = _extract("car-rental-services.pdf", "Services Breakdown")

    assert extracted.page_count == 1
    assert "Economy & Sedan Rental" in extracted.text
    assert "Chauffeur" in extracted.text


def test_policies_pdf_extracts_expected_content():
    extracted = _extract("car-rental-policies.pdf", "Terms & Rental Policies")

    assert extracted.page_count == 2
    assert "Minimum Age" in extracted.text
    assert "Late Return Policy" in extracted.text
