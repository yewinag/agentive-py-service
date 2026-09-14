import asyncio

from app.knowledge.fake_extractor import FakeDocumentExtractor
from app.knowledge.models import DocumentSource, ExtractedDocument


def test_fake_extractor_returns_extracted_document():
    extractor = FakeDocumentExtractor()
    source = DocumentSource(
        id="doc-1", title="Car Rental Policy", content="  Some policy text.  "
    )

    extracted = asyncio.run(extractor.extract(source))

    assert extracted.document_id == "doc-1"
    assert extracted.title == "Car Rental Policy"
    assert extracted.text == "Some policy text."


async def _extract_with(extractor, source):
    return await extractor.extract(source)


def test_extraction_works_with_any_object_satisfying_the_protocol():
    class CustomExtractor:
        """Unrelated to FakeDocumentExtractor - proves the Protocol,
        not a concrete class, is what callers actually depend on."""

        async def extract(self, source: DocumentSource) -> ExtractedDocument:
            return ExtractedDocument(
                document_id=source.id,
                title=source.title,
                text=f"custom:{source.content}",
            )

    source = DocumentSource(id="doc-2", title="Services", content="raw")

    extracted = asyncio.run(_extract_with(CustomExtractor(), source))

    assert extracted.text == "custom:raw"
