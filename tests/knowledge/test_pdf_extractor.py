import asyncio

import pytest

from app.knowledge.exceptions import DocumentExtractionError
from app.knowledge.models import DocumentSource
from app.knowledge.pdf_extractor import PdfDocumentExtractor
from tests.knowledge.pdf_fixtures import build_minimal_pdf


def test_extract_returns_text_and_page_count(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(build_minimal_pdf("Hello PDF"))
    source = DocumentSource(id="doc-1", title="Sample", content=str(pdf_path))
    extractor = PdfDocumentExtractor()

    extracted = asyncio.run(extractor.extract(source))

    assert extracted.document_id == "doc-1"
    assert extracted.title == "Sample"
    assert "Hello PDF" in extracted.text
    assert extracted.page_count == 1


def test_extract_raises_document_extraction_error_for_invalid_pdf(tmp_path):
    pdf_path = tmp_path / "not-a-pdf.pdf"
    pdf_path.write_bytes(b"this is not a valid pdf file")
    source = DocumentSource(id="doc-2", title="Broken", content=str(pdf_path))
    extractor = PdfDocumentExtractor()

    with pytest.raises(DocumentExtractionError):
        asyncio.run(extractor.extract(source))


def test_extract_raises_document_extraction_error_for_missing_file(tmp_path):
    missing_path = tmp_path / "does-not-exist.pdf"
    source = DocumentSource(id="doc-3", title="Missing", content=str(missing_path))
    extractor = PdfDocumentExtractor()

    with pytest.raises(DocumentExtractionError):
        asyncio.run(extractor.extract(source))
