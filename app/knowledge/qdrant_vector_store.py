import uuid
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.knowledge.exceptions import VectorStoreError
from app.knowledge.models import DocumentChunk
from app.knowledge.vector_store import VectorRecord, VectorSearchResult

DEFAULT_COLLECTION_NAME = "knowledge_chunk_embeddings"

# Qdrant point ids must be an unsigned integer or a UUID - arbitrary
# strings (this project's chunk ids, e.g. "01-rental-services.pdf-chunk-0")
# are rejected by the server. This namespace makes a deterministic UUID
# out of any chunk id, so the same chunk id always maps to the same
# point (required for add() to upsert rather than duplicate) without a
# separate id-mapping table. The original chunk id is kept in the
# payload and is what callers actually see back from search().
_POINT_ID_NAMESPACE = uuid.UUID("6a3f2b1e-2f0a-4b1a-9d8c-6e6f2a6f5b3d")


class QdrantVectorStore:
    """VectorStore implementation backed by Qdrant. The qdrant-client
    library is an implementation detail of this module alone - nothing
    outside app/knowledge ever imports it, the same isolation
    PgVectorStore already applies to SQLAlchemy/pgvector.

    Uses cosine distance, matching InMemoryVectorStore's cosine
    similarity and PgVectorStore's choice - all three VectorStore
    implementations rank results the same way regardless of backend.
    Qdrant's query score for Cosine distance already is the similarity
    itself (higher = more similar), so - unlike PgVectorStore, which
    converts a distance to a similarity - no conversion is needed here.
    """

    def __init__(
        self,
        url: str,
        dimensions: int,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client: Optional[AsyncQdrantClient] = None,
    ) -> None:
        self._client = client or AsyncQdrantClient(url=url)
        self._collection_name = collection_name
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        """The fixed vector width this store's collection was created
        for. Lets a caller (or a test) confirm the store agrees with
        whatever EmbeddingProvider is about to write to it.
        """
        return self._dimensions

    async def ensure_collection(self) -> None:
        """Creates this store's collection if it doesn't already exist.
        Not called automatically by __init__: collection setup is an
        explicit, one-time operation, the same reasoning PgVectorStore's
        create_schema() documents - not something that should happen
        implicitly on every app start.
        """
        try:
            if await self._client.collection_exists(self._collection_name):
                return
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=self._dimensions, distance=Distance.COSINE),
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to ensure Qdrant collection: {exc}") from exc

    async def reset_collection(self) -> None:
        """Deletes this store's collection (if it exists) and recreates
        it empty. The simplest safe way to guarantee no stale vectors
        survive for a document that has been renamed or removed from the
        canonical set - a full rebuild, not an incremental sync. See the
        ingestion command's --reset flag and README's "Handling stale
        documents" section for the tradeoff this accepts.
        """
        try:
            if await self._client.collection_exists(self._collection_name):
                await self._client.delete_collection(self._collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Failed to reset Qdrant collection: {exc}") from exc
        await self.ensure_collection()

    async def delete_collection(self) -> None:
        """Drops this store's collection. The inverse of
        ensure_collection() - exists for test/dev teardown, not used by
        the application itself.
        """
        try:
            await self._client.delete_collection(self._collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete Qdrant collection: {exc}") from exc

    async def close(self) -> None:
        """Closes the underlying client connection. Call during test/dev
        teardown to avoid leaking connections across test runs.
        """
        await self._client.close()

    async def add(self, records: list[VectorRecord]) -> None:
        if not records:
            return

        points = [
            PointStruct(
                id=_point_id_for(record.chunk.id),
                vector=record.embedding,
                payload=_chunk_to_payload(record.chunk),
            )
            for record in records
        ]

        try:
            await self._client.upsert(collection_name=self._collection_name, points=points)
        except Exception as exc:
            raise VectorStoreError(f"Failed to store vector records: {exc}") from exc

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[VectorSearchResult]:
        try:
            if not await self._client.collection_exists(self._collection_name):
                return []

            response = await self._client.query_points(
                collection_name=self._collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to search vector records: {exc}") from exc

        return [_point_to_result(point) for point in response.points]


def _point_id_for(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def _chunk_to_payload(chunk: DocumentChunk) -> dict:
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "document_title": chunk.document_title,
        "section_heading": chunk.section_heading,
        "text": chunk.text,
        "position": chunk.position,
    }


def _point_to_result(point) -> VectorSearchResult:
    payload = point.payload
    chunk = DocumentChunk(
        id=payload["chunk_id"],
        document_id=payload["document_id"],
        document_title=payload["document_title"],
        section_heading=payload["section_heading"],
        text=payload["text"],
        position=payload["position"],
    )
    return VectorSearchResult(chunk=chunk, score=point.score)
