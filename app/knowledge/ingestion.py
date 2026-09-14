from app.core.config import Settings
from app.knowledge.chunking import DocumentChunker
from app.knowledge.embedding import EmbeddingProvider
from app.knowledge.extraction import DocumentExtractor
from app.knowledge.models import DocumentSource
from app.knowledge.vector_store import VectorRecord, VectorStore


class IngestionService:
    """Composes the existing knowledge-pipeline Protocols
    (DocumentExtractor, DocumentChunker, EmbeddingProvider, VectorStore)
    into one operation: turn raw DocumentSources into searchable,
    embedded, stored chunks. Reimplements none of them - it only
    sequences calls to abstractions that are already tested on their own.

    Deliberately not exposed as an HTTP endpoint: ingestion is an
    occasional, batch operation (run when the knowledge base changes),
    not a per-request one. No file upload, admin CRUD, or scheduling -
    those are separate future concerns, not part of this foundation.
    """

    def __init__(
        self,
        extractor: DocumentExtractor,
        chunker: DocumentChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._extractor = extractor
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def ingest(self, sources: list[DocumentSource]) -> int:
        """Extracts, chunks, embeds, and stores every source. Returns
        the total number of chunks stored.
        """
        total_chunks = 0

        for source in sources:
            extracted = await self._extractor.extract(source)
            chunks = self._chunker.chunk(extracted)
            if not chunks:
                continue

            embeddings = await self._embedding_provider.embed(
                [chunk.text for chunk in chunks]
            )
            records = [
                VectorRecord(chunk=chunk, embedding=embedding)
                for chunk, embedding in zip(chunks, embeddings)
            ]
            await self._vector_store.add(records)
            total_chunks += len(records)

        return total_chunks


def get_ingestion_service(settings: Settings) -> IngestionService:
    """Composition point - a plain function, not FastAPI Depends()-wired,
    matching every other composition point in app/knowledge. Extraction
    and chunking aren't Settings-selected like the other layers: there is
    only one real implementation of each that makes sense for ingesting
    actual documents (PdfDocumentExtractor, SectionAwareChunker) - the
    Fake variants exist purely for tests, never for real ingestion.
    """
    from app.knowledge.chunking import SectionAwareChunker
    from app.knowledge.embedding import get_embedding_provider
    from app.knowledge.pdf_extractor import PdfDocumentExtractor
    from app.knowledge.vector_store import get_vector_store

    return IngestionService(
        extractor=PdfDocumentExtractor(),
        chunker=SectionAwareChunker(),
        embedding_provider=get_embedding_provider(settings),
        vector_store=get_vector_store(settings),
    )
