import pdfplumber

from app.knowledge.exceptions import DocumentExtractionError
from app.knowledge.models import DocumentSource, ExtractedDocument


class PdfDocumentExtractor:
    """DocumentExtractor implementation backed by pdfplumber. pdfplumber
    (and PDF parsing in general) is an implementation detail of this
    module alone - nothing outside app/knowledge ever imports it.

    `source.content` is treated as a filesystem path to the PDF file.

    PDF parsing is CPU-bound, not I/O-bound, so no threads/executors are
    used here - `extract` just does the blocking work directly. It's
    still declared async to satisfy DocumentExtractor: every extractor is
    called the same way, and a future one (e.g. an OCR API for scanned
    PDFs) will do genuine I/O and need that async signature for real.
    """

    async def extract(self, source: DocumentSource) -> ExtractedDocument:
        try:
            with pdfplumber.open(source.content) as pdf:
                page_texts = [page.extract_text() or "" for page in pdf.pages]
        except Exception as exc:
            # pdfplumber/pdfminer don't expose one stable exception type
            # for "this PDF is invalid" (syntax errors, missing files,
            # decode failures all surface differently), so this boundary
            # translates anything that goes wrong into our own exception
            # rather than leaking a library-specific one.
            raise DocumentExtractionError(
                f"Failed to extract PDF for document '{source.id}': {exc}"
            ) from exc

        text = "\n\n".join(page_text.strip() for page_text in page_texts if page_text.strip())
        if not text:
            raise DocumentExtractionError(
                f"PDF for document '{source.id}' contained no extractable text"
            )

        return ExtractedDocument(
            document_id=source.id,
            title=source.title,
            text=text,
            page_count=len(page_texts),
        )
