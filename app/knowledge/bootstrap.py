from pathlib import Path

from app.core.config import Settings
from app.knowledge.ingestion import get_ingestion_service
from app.knowledge.models import DocumentSource

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"

# The six canonical documents this project has today (see
# data/knowledge/). A hardcoded mapping is honest about that, rather
# than deriving a title from the filename for documents that don't
# exist yet. The two older, superseded PDFs (car-rental-services.pdf,
# car-rental-policies.pdf) have been retired in favor of these.
_KNOWN_DOCUMENT_TITLES = {
    "01-rental-services.pdf": "Rental Services",
    "02-rental-policies.pdf": "Rental Policies",
    "03-booking-policy.pdf": "Booking Policy",
    "04-cancellation-policy.pdf": "Cancellation Policy",
    "05-payment-policy.pdf": "Payment Policy",
    "06-pickup-return-policy.pdf": "Pickup and Return Policy",
}


def discover_canonical_sources() -> list[DocumentSource]:
    """Scans data/knowledge/ for the project's PDFs and returns them as
    DocumentSources, with clean titles for the known canonical ones.
    "What documents exist" has exactly one answer; both
    bootstrap_default_knowledge_base() below (the in-memory startup
    path) and app/knowledge/ingest.py (the explicit Qdrant ingestion
    command) ask it here rather than each re-implementing the scan.
    """
    return [
        DocumentSource(
            id=path.name,
            title=_KNOWN_DOCUMENT_TITLES.get(path.name, path.stem),
            content=str(path),
        )
        for path in sorted(KNOWLEDGE_DIR.glob("*.pdf"))
    ]


async def bootstrap_default_knowledge_base(settings: Settings) -> int:
    """Ingests the project's committed PDFs into the configured
    VectorStore. Reuses IngestionService - no ingestion logic lives here.

    Only runs for vector_store_provider == "memory". That store starts
    empty every process start (it's a process-wide, in-process
    singleton - see get_vector_store()), so without this, the default
    local/dev configuration would always answer "not available" even
    though the knowledge pipeline itself works. A persistent store
    (pgvector, qdrant) is a real, already-shared datastore: ingesting
    into it is a deliberate, occasional operation - see
    app/knowledge/ingest.py for Qdrant's explicit ingestion command -
    not something to silently redo on every app boot, which would waste
    real embedding API calls for no benefit, since the data is already
    there.

    Idempotent: VectorStore.add() is upsert-by-chunk-id, and this is
    called at most once per process (from app startup), so repeated
    calls (or app restarts) never duplicate stored chunks.
    """
    if settings.vector_store_provider != "memory":
        return 0

    sources = discover_canonical_sources()
    if not sources:
        return 0

    ingestion_service = get_ingestion_service(settings)
    return await ingestion_service.ingest(sources)
