import asyncio

from app.knowledge.chunking import SectionAwareChunker
from app.knowledge.fake_embedding_provider import FakeEmbeddingProvider
from app.knowledge.fake_extractor import FakeDocumentExtractor
from app.knowledge.in_memory_vector_store import InMemoryVectorStore
from app.knowledge.ingestion import IngestionService
from app.knowledge.models import DocumentSource

_ZERO_QUERY = [0.0] * 8  # FakeEmbeddingProvider's default dimensionality


def _service(vector_store=None):
    return IngestionService(
        extractor=FakeDocumentExtractor(),
        chunker=SectionAwareChunker(),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store or InMemoryVectorStore(),
    )


def _source(source_id: str, content: str) -> DocumentSource:
    return DocumentSource(id=source_id, title=f"Title {source_id}", content=content)


def _all_stored(store: InMemoryVectorStore, limit: int = 50):
    # A zero query vector ties every record at score 0.0 (stable sort),
    # which is a convenient public-API way to inspect "everything stored"
    # without reaching into the store's private state.
    return asyncio.run(store.search(query_embedding=_ZERO_QUERY, top_k=limit))


def test_ingest_stores_all_chunks_from_a_document():
    store = InMemoryVectorStore()
    service = _service(store)
    source = _source(
        "doc-1",
        "1. Section One\n● Item one detail.\n2. Section Two\n● Item two detail.",
    )

    count = asyncio.run(service.ingest([source]))

    assert count == 2  # two numbered sections, no preamble text before "1."
    assert len(_all_stored(store)) == count


def test_ingest_stores_embeddings_with_correct_chunk_metadata():
    store = InMemoryVectorStore()
    service = _service(store)
    source = _source("policies", "1. Driver Eligibility\n● Minimum Age: 21.")

    asyncio.run(service.ingest([source]))

    [result] = _all_stored(store)
    chunk = result.chunk
    assert chunk.document_id == "policies"
    assert chunk.document_title == "Title policies"
    assert chunk.section_heading == "1. Driver Eligibility"
    assert "Minimum Age" in chunk.text


def test_ingest_embeds_and_makes_chunks_findable_by_their_own_text():
    store = InMemoryVectorStore()
    service = _service(store)
    source = _source("policies", "1. Driver Eligibility\n● Minimum Age: 21.")
    asyncio.run(service.ingest([source]))

    embedding_provider = FakeEmbeddingProvider()
    stored_text = _all_stored(store)[0].chunk.text
    [query_embedding] = asyncio.run(embedding_provider.embed([stored_text]))

    results = asyncio.run(store.search(query_embedding=query_embedding, top_k=1))

    assert results[0].score > 0.99


def test_ingest_handles_multiple_sources():
    store = InMemoryVectorStore()
    service = _service(store)
    sources = [
        _source("doc-a", "1. Section A\n● Detail A."),
        _source("doc-b", "1. Section B\n● Detail B."),
    ]

    count = asyncio.run(service.ingest(sources))

    document_ids = {result.chunk.document_id for result in _all_stored(store)}
    assert document_ids == {"doc-a", "doc-b"}
    assert count == len(_all_stored(store))


def test_ingest_on_empty_source_list_is_a_no_op():
    store = InMemoryVectorStore()
    service = _service(store)

    count = asyncio.run(service.ingest([]))

    assert count == 0
    assert _all_stored(store) == []
