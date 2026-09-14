from pydantic import BaseModel, Field


class DocumentSource(BaseModel):
    """A raw, not-yet-extracted document entering the pipeline.

    Today `content` models fake, already-textual input. A real source
    (a PDF's bytes, a file path) will likely change this field's type,
    but ExtractedDocument/DocumentChunk downstream won't need to change.
    """

    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class ExtractedDocument(BaseModel):
    """Clean text pulled out of a DocumentSource, ready to be chunked."""

    document_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class DocumentChunk(BaseModel):
    """One retrievable unit of an ExtractedDocument.

    The chunking algorithm itself isn't implemented yet - this is the
    shape a future chunker, embedder, and retriever will all agree on.
    """

    id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    position: int = Field(..., ge=0)
