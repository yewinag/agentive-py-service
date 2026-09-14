import pytest
from pydantic import ValidationError

from app.knowledge.models import DocumentChunk, DocumentSource, ExtractedDocument


def test_document_source_rejects_empty_content():
    with pytest.raises(ValidationError):
        DocumentSource(id="doc-1", title="Title", content="")


def test_extracted_document_rejects_empty_text():
    with pytest.raises(ValidationError):
        ExtractedDocument(document_id="doc-1", title="Title", text="")


def test_document_chunk_rejects_negative_position():
    with pytest.raises(ValidationError):
        DocumentChunk(
            id="chunk-1",
            document_id="doc-1",
            document_title="Title",
            text="chunk text",
            position=-1,
        )


def test_document_chunk_accepts_valid_data():
    chunk = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        document_title="Title",
        section_heading="1. Intro",
        text="chunk text",
        position=0,
    )

    assert chunk.position == 0
    assert chunk.section_heading == "1. Intro"
