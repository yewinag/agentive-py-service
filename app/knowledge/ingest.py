"""Explicit, repeatable knowledge ingestion command.

    python -m app.knowledge.ingest [--reset]

Ingests the project's canonical PDFs (data/knowledge/) into Qdrant.
Unlike bootstrap_default_knowledge_base() (which only ever populates the
transient in-memory store, automatically, once per process, from inside
the FastAPI app's startup), this is a standalone CLI: no FastAPI app, no
web server - run deliberately, only when the knowledge base actually
changes, exactly the "ingestion is an occasional operation" reasoning
IngestionService and bootstrap.py already document.

Requires VECTOR_STORE_PROVIDER=qdrant - this command exists specifically
to persist knowledge into Qdrant, not to duplicate bootstrap's in-memory
path or add a generic multi-backend ingestion tool.
"""
import argparse
import asyncio
import sys

from app.core.config import Settings, get_settings
from app.knowledge.bootstrap import discover_canonical_sources
from app.knowledge.chunking import SectionAwareChunker
from app.knowledge.embedding import get_embedding_provider
from app.knowledge.exceptions import DocumentExtractionError, EmbeddingProviderError, VectorStoreError
from app.knowledge.ingestion import IngestionService
from app.knowledge.pdf_extractor import PdfDocumentExtractor
from app.knowledge.vector_store import get_vector_store


async def run(settings: Settings, reset: bool) -> int:
    if settings.vector_store_provider != "qdrant":
        print(
            f"VECTOR_STORE_PROVIDER is '{settings.vector_store_provider}', not 'qdrant'. "
            "This command persists knowledge into Qdrant - set VECTOR_STORE_PROVIDER=qdrant "
            "and re-run.",
            file=sys.stderr,
        )
        return 1

    print("Knowledge ingestion started")

    sources = discover_canonical_sources()
    print(f"Documents discovered: {len(sources)}")
    if not sources:
        print("No PDFs found in data/knowledge/ - nothing to ingest.")
        print("Status: SUCCESS")
        return 0

    vector_store = get_vector_store(settings)
    try:
        if reset:
            print(f"Resetting collection '{settings.qdrant_collection}' (--reset)...")
            await vector_store.reset_collection()
        else:
            await vector_store.ensure_collection()
    except VectorStoreError as exc:
        print(f"Could not prepare Qdrant collection at {settings.qdrant_url}: {exc}", file=sys.stderr)
        print("Status: FAILED")
        return 1

    ingestion = IngestionService(
        extractor=PdfDocumentExtractor(),
        chunker=SectionAwareChunker(),
        embedding_provider=get_embedding_provider(settings),
        vector_store=vector_store,
    )

    try:
        chunk_count = await ingestion.ingest(sources)
    except (DocumentExtractionError, EmbeddingProviderError, VectorStoreError) as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        print("Status: FAILED")
        return 1
    finally:
        await vector_store.close()

    print(f"Documents processed: {len(sources)}")
    print(f"Chunks generated: {chunk_count}")
    print(f"Vectors upserted: {chunk_count}")
    print(f"Collection: {settings.qdrant_collection}")
    print("Status: SUCCESS")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the canonical knowledge PDFs (data/knowledge/) into Qdrant."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete and recreate the collection before ingesting. Use when a canonical PDF "
            "has been renamed or removed, to avoid leaving its old vectors orphaned - see "
            "README's 'Handling stale documents' section."
        ),
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run(get_settings(), reset=args.reset)))


if __name__ == "__main__":
    main()
