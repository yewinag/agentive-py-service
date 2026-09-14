from typing import Protocol

from app.knowledge.models import DocumentSource, ExtractedDocument


class DocumentExtractor(Protocol):
    """The boundary a future ingestion pipeline codes against: WHAT it
    needs to turn a DocumentSource into clean text. Concrete extractors
    (fake today, a real PDF-backed one later) implement this structurally.
    """

    async def extract(self, source: DocumentSource) -> ExtractedDocument: ...
