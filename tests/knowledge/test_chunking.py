from app.knowledge.chunking import SectionAwareChunker
from app.knowledge.models import ExtractedDocument

TWO_SECTION_TEXT = (
    "Document 9: Sample Policy\n"
    "1. Driver Eligibility & Required Documents\n"
    "● Minimum Age: Renters must be at least 21 years old.\n"
    "● Driving Experience: Must hold a valid license for at least 1 year.\n"
    "2. Payment & Security Deposit\n"
    "● Accepted Payment Methods: Credit Card, Bank Transfer, or PromptPay.\n"
    "● Security Deposit: A refundable deposit is required upon handover.\n"
)


def _document(text: str = TWO_SECTION_TEXT) -> ExtractedDocument:
    return ExtractedDocument(document_id="doc-9", title="Sample Policy", text=text, page_count=1)


def test_chunks_one_section_per_numbered_heading():
    chunker = SectionAwareChunker()

    chunks = chunker.chunk(_document())

    headings = [c.section_heading for c in chunks]
    assert "1. Driver Eligibility & Required Documents" in headings
    assert "2. Payment & Security Deposit" in headings


def test_preamble_before_first_heading_becomes_its_own_chunk():
    chunker = SectionAwareChunker()

    chunks = chunker.chunk(_document())

    assert chunks[0].section_heading is None
    assert "Document 9: Sample Policy" in chunks[0].text


def test_section_heading_is_preserved_inline_in_chunk_text():
    chunker = SectionAwareChunker()

    chunks = chunker.chunk(_document())
    deposit_chunk = next(c for c in chunks if c.section_heading == "2. Payment & Security Deposit")

    assert deposit_chunk.text.startswith("2. Payment & Security Deposit")
    assert "Security Deposit: A refundable deposit" in deposit_chunk.text


def test_chunk_metadata_preserves_document_identity_and_position():
    chunker = SectionAwareChunker()

    chunks = chunker.chunk(_document())

    for expected_position, chunk in enumerate(chunks):
        assert chunk.document_id == "doc-9"
        assert chunk.document_title == "Sample Policy"
        assert chunk.position == expected_position
        assert chunk.id == f"doc-9-chunk-{expected_position}"


def test_chunking_is_deterministic():
    chunker = SectionAwareChunker()
    document = _document()

    first_run = chunker.chunk(document)
    second_run = chunker.chunk(document)

    assert first_run == second_run


def test_chunking_does_not_mutate_the_original_document():
    document = _document()
    original_text = document.text

    SectionAwareChunker().chunk(document)

    assert document.text == original_text


def test_long_section_is_split_with_heading_repeated_and_overlap():
    long_section_text = (
        "1. Long Section\n"
        + "".join(f"● Item {i}: {'x' * 60}\n" for i in range(5))
    )
    document = ExtractedDocument(
        document_id="doc-10", title="Long Doc", text=long_section_text, page_count=1
    )
    chunker = SectionAwareChunker(max_chunk_chars=150, overlap_chars=20)

    chunks = chunker.chunk(document)

    assert len(chunks) > 1
    assert all(c.section_heading == "1. Long Section" for c in chunks)
    assert all(c.text.startswith("1. Long Section") for c in chunks)
    # every window after the first should carry a trailing slice of the
    # previous window's text for continuity
    for previous, current in zip(chunks, chunks[1:]):
        tail = previous.text[-20:]
        assert tail in current.text
