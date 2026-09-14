from pydantic import BaseModel, Field


class DocumentSource(BaseModel):
    """A raw, not-yet-extracted document entering the pipeline.

    `content` is interpreted by whichever DocumentExtractor consumes it:
    FakeDocumentExtractor treats it as literal inline text;
    PdfDocumentExtractor treats it as a filesystem path to a PDF. A
    plain str turns out to be enough for both cases (a path is still a
    string), so the field's shape didn't need to change to support PDFs.
    """

    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class ExtractedDocument(BaseModel):
    """Clean text pulled out of a DocumentSource, ready to be chunked."""

    document_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    page_count: int = Field(default=1, ge=1)


class DocumentChunk(BaseModel):
    """One retrievable unit of an ExtractedDocument.

    The chunking algorithm itself isn't implemented yet - this is the
    shape a future chunker, embedder, and retriever will all agree on.
    """

    id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    position: int = Field(..., ge=0)
