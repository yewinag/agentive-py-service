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


async def bootstrap_default_knowledge_base(settings: Settings) -> int:
    """Ingests the project's committed PDFs into the configured
    VectorStore. Reuses IngestionService - no ingestion logic lives here.

    Only runs for vector_store_provider == "memory". That store starts
    empty every process start (it's a process-wide, in-process
    singleton - see get_vector_store()), so without this, the default
    local/dev configuration would always answer "not available" even
    though the knowledge pipeline itself works. A pgvector store is a
    real, persistent, already-shared database: ingesting into it is a
    deliberate, occasional operation (see IngestionService's own docs),
    not something to silently redo on every app boot - doing so would
    waste real embedding API calls for no benefit, since the data is
    already there.

    Idempotent: VectorStore.add() is upsert-by-chunk-id, and this is
    called at most once per process (from app startup), so repeated
    calls (or app restarts) never duplicate stored chunks.
    """
    if settings.vector_store_provider != "memory":
        return 0

    sources = [
        DocumentSource(
            id=path.name,
            title=_KNOWN_DOCUMENT_TITLES.get(path.name, path.stem),
            content=str(path),
        )
        for path in sorted(KNOWLEDGE_DIR.glob("*.pdf"))
    ]
    if not sources:
        return 0

    ingestion_service = get_ingestion_service(settings)
    return await ingestion_service.ingest(sources)
