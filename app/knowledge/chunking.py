import re
from typing import Optional, Protocol

from app.knowledge.models import DocumentChunk, ExtractedDocument

# Justified by the real documents, not chosen arbitrarily: the two
# car-rental PDFs' numbered sections range 329-754 chars (see README /
# Step 7 analysis). max_chunk_chars gives ~1.5-2x headroom above the
# largest real section so every real section fits as a single chunk,
# while still being a real, enforced bound for future, larger documents.
DEFAULT_MAX_CHUNK_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 150

_SECTION_HEADING_RE = re.compile(r"^\d+\.\s+\S")
_BULLET_RE = re.compile(r"^[●•]\s")  # '●' and '•'


class DocumentChunker(Protocol):
    """The boundary a future ingestion pipeline codes against: WHAT it
    needs to turn an ExtractedDocument into retrievable chunks.
    """

    def chunk(self, document: ExtractedDocument) -> list[DocumentChunk]: ...


class SectionAwareChunker:
    """Splits on the documents' own numbered section headings ("1. ...",
    "2. ..."), keeping each section as one chunk with its heading
    prepended so a chunk is understandable on its own once retrieved.

    A section longer than max_chunk_chars is split further along bullet
    boundaries, with a small character overlap so context isn't lost at
    the cut. Overlap is only ever applied *within* one section's split,
    never between two different sections - carrying the tail of a
    "Cancellation Policy" chunk into a "Vehicle Use Rules" chunk would
    blend unrelated topics and hurt retrieval precision, not help it.

    Pure in-memory text processing: no I/O, no external library, so
    (unlike LLMProvider/DocumentExtractor) there's no expensive or flaky
    dependency worth faking here - calling this class directly in tests
    already is fast and deterministic.
    """

    def __init__(
        self,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    ) -> None:
        self._max_chunk_chars = max_chunk_chars
        self._overlap_chars = overlap_chars

    def chunk(self, document: ExtractedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        for heading, body in _split_into_sections(document.text):
            full_text = f"{heading}\n{body}" if heading else body
            if not full_text.strip():
                continue

            if len(full_text) <= self._max_chunk_chars:
                texts = [full_text]
            else:
                texts = _split_long_section(
                    heading, body, self._max_chunk_chars, self._overlap_chars
                )

            for text in texts:
                chunks.append(
                    DocumentChunk(
                        id=f"{document.document_id}-chunk-{len(chunks)}",
                        document_id=document.document_id,
                        document_title=document.title,
                        section_heading=heading,
                        text=text.strip(),
                        position=len(chunks),
                    )
                )

        return chunks


def _split_into_sections(text: str) -> list:
    """Groups text into (heading, body) pairs on lines matching the
    documents' numbered-heading style ("1. Driver Eligibility ...").
    `heading` is None only for text preceding the first recognized
    heading (or for a document with no recognized headings at all).
    """
    sections = []
    heading: Optional[str] = None
    body_lines: list = []

    for line in text.split("\n"):
        if _SECTION_HEADING_RE.match(line.strip()):
            if heading is not None or body_lines:
                sections.append((heading, "\n".join(l for l in body_lines if l.strip())))
            heading = line.strip()
            body_lines = []
        else:
            body_lines.append(line)

    if heading is not None or body_lines:
        sections.append((heading, "\n".join(l for l in body_lines if l.strip())))

    return sections


def _split_into_items(body: str) -> list:
    """Groups a section body's lines into bullet-item units (a bullet
    line plus its wrapped continuation lines). Falls back to one item
    per line when the body has no bullets to split on.
    """
    lines = [line for line in body.split("\n") if line.strip()]
    if not any(_BULLET_RE.match(line) for line in lines):
        return lines

    items: list = []
    current: list = []
    for line in lines:
        if _BULLET_RE.match(line) and current:
            items.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        items.append("\n".join(current))
    return items


def _split_long_section(
    heading: Optional[str], body: str, max_chunk_chars: int, overlap_chars: int
) -> list:
    """Packs a section's bullet items into windows no larger than
    max_chunk_chars, each re-prefixed with the section heading and a
    trailing slice of the previous window for continuity. A single item
    longer than max_chunk_chars is kept whole rather than cut mid-line.
    """
    items = _split_into_items(body)
    prefix = f"{heading}\n" if heading else ""

    windows: list = []
    current_items: list = []
    leading_overlap = ""

    for item in items:
        tentative_items = current_items + [item]
        tentative_text = prefix + leading_overlap + "\n".join(tentative_items)

        if current_items and len(tentative_text) > max_chunk_chars:
            finished_text = prefix + leading_overlap + "\n".join(current_items)
            windows.append(finished_text)
            leading_overlap = finished_text[-overlap_chars:] + "\n" if overlap_chars else ""
            current_items = [item]
        else:
            current_items = tentative_items

    if current_items:
        windows.append(prefix + leading_overlap + "\n".join(current_items))

    return windows
