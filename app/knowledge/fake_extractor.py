from app.knowledge.models import DocumentSource, ExtractedDocument


class FakeDocumentExtractor:
    """Deterministic stand-in for a real extractor (e.g. PDF text
    extraction). Treats the source's content as already-clean text.
    """

    async def extract(self, source: DocumentSource) -> ExtractedDocument:
        return ExtractedDocument(
            document_id=source.id,
            title=source.title,
            text=source.content.strip(),
        )
