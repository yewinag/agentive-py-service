from typing import Optional

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.knowledge.exceptions import VectorStoreError
from app.knowledge.models import DocumentChunk
from app.knowledge.vector_store import VectorRecord, VectorSearchResult

DEFAULT_TABLE_NAME = "knowledge_chunk_embeddings"


class PgVectorStore:
    """VectorStore implementation backed by PostgreSQL + pgvector.
    SQLAlchemy and the pgvector/asyncpg libraries are implementation
    details of this module alone - nothing outside app/knowledge ever
    imports them.

    The table is built with SQLAlchemy Core (not a declarative ORM
    model) because its vector column's width depends on `dimensions`, a
    runtime configuration value (whichever embedding model is
    selected) - not something fixable at class-definition time.

    Uses cosine distance (pgvector's `<=>` operator), matching
    InMemoryVectorStore's cosine similarity and appropriate for OpenAI
    embeddings, which are documented as unit-normalized.
    """

    def __init__(
        self,
        database_url: str,
        dimensions: int,
        table_name: str = DEFAULT_TABLE_NAME,
        engine: Optional[AsyncEngine] = None,
    ) -> None:
        self._engine = engine or create_async_engine(database_url)
        self._dimensions = dimensions
        self._metadata = sa.MetaData()
        self._table = sa.Table(
            table_name,
            self._metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("document_id", sa.String, nullable=False, index=True),
            sa.Column("document_title", sa.String, nullable=False),
            sa.Column("section_heading", sa.String, nullable=True),
            sa.Column("text", sa.Text, nullable=False),
            sa.Column("position", sa.Integer, nullable=False),
            sa.Column("embedding", Vector(dimensions), nullable=False),
        )

    @property
    def dimensions(self) -> int:
        """The fixed vector width this store's table was created for.
        Lets a caller (or a test) confirm the store agrees with whatever
        EmbeddingProvider is about to write to it, without reaching into
        SQLAlchemy internals.
        """
        return self._dimensions

    async def create_schema(self) -> None:
        """Creates the pgvector extension and this store's table if they
        don't already exist yet. Not called automatically by __init__:
        schema setup is an explicit, one-time operation, not something
        that should happen implicitly on every app start.
        """
        try:
            async with self._engine.begin() as conn:
                await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.run_sync(self._metadata.create_all)
        except Exception as exc:
            raise VectorStoreError(f"Failed to create pgvector schema: {exc}") from exc

    async def drop_schema(self) -> None:
        """Drops this store's table. The inverse of create_schema() -
        exists for test/dev teardown, not used by the application itself.
        """
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(self._metadata.drop_all)
        except Exception as exc:
            raise VectorStoreError(f"Failed to drop pgvector schema: {exc}") from exc

    async def dispose(self) -> None:
        """Closes the underlying connection pool. Call during test/dev
        teardown to avoid leaking connections across test runs.
        """
        await self._engine.dispose()

    async def add(self, records: list[VectorRecord]) -> None:
        if not records:
            return

        rows = [_record_to_row(record) for record in records]
        statement = pg_insert(self._table).values(rows)
        statement = statement.on_conflict_do_update(
            index_elements=["id"],
            set_={
                column: getattr(statement.excluded, column)
                for column in rows[0]
                if column != "id"
            },
        )

        try:
            async with self._engine.begin() as conn:
                await conn.execute(statement)
        except Exception as exc:
            raise VectorStoreError(f"Failed to store vector records: {exc}") from exc

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[VectorSearchResult]:
        distance = self._table.c.embedding.cosine_distance(query_embedding).label("distance")
        statement = sa.select(self._table, distance).order_by(distance).limit(top_k)

        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(statement)
                rows = result.mappings().all()
        except Exception as exc:
            raise VectorStoreError(f"Failed to search vector records: {exc}") from exc

        return [_row_to_result(row) for row in rows]


def _record_to_row(record: VectorRecord) -> dict:
    return {
        "id": record.chunk.id,
        "document_id": record.chunk.document_id,
        "document_title": record.chunk.document_title,
        "section_heading": record.chunk.section_heading,
        "text": record.chunk.text,
        "position": record.chunk.position,
        "embedding": record.embedding,
    }


def _row_to_result(row) -> VectorSearchResult:
    chunk = DocumentChunk(
        id=row["id"],
        document_id=row["document_id"],
        document_title=row["document_title"],
        section_heading=row["section_heading"],
        text=row["text"],
        position=row["position"],
    )
    # pgvector's `<=>` is cosine DISTANCE (0 = identical); convert to
    # similarity so both VectorStore implementations share one "higher is
    # more similar" score contract regardless of backend.
    return VectorSearchResult(chunk=chunk, score=1.0 - row["distance"])
