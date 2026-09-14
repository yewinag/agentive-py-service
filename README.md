# Agentive AI Service

The Python/FastAPI AI service for the Agentive Car Rental platform. It is intentionally
separated from the main business/backend service (NestJS): the business API owns bookings,
users, and core domain logic, while this service owns everything AI-specific — LLM
interaction, knowledge retrieval, and (eventually) agent orchestration and tool calling.

## Architecture overview

```
Frontend
  ↓
Main Business API
  ↓
FastAPI Agentive Service
  ├── Chat            (app/chats)
  ├── LLM Provider     (app/llm)
  └── Knowledge Pipeline (app/knowledge)
```

Three boundaries, each with a single responsibility:

- **Chat/application layer** (`app/chats`) — orchestrates one request: validates input,
  calls the layers below it, shapes the response. No LLM- or document-specific logic lives
  here.
- **LLM provider layer** (`app/llm`) — answers "how do we generate a reply?" behind an
  `LLMProvider` Protocol, so the concrete provider (a fake, OpenAI, or anything else later)
  is swappable without touching the chat layer.
- **Knowledge/document layer** (`app/knowledge`) — turns source documents into retrievable,
  embeddable, storable units: extraction (`DocumentExtractor`), chunking (`DocumentChunker`),
  embedding (`EmbeddingProvider`), and vector storage (`VectorStore`), each behind its own
  Protocol. Framework-independent: it doesn't import FastAPI and isn't wired into any
  endpoint yet.

## Current implementation status

**Implemented:**
- FastAPI application structure (app factory, feature-package layout)
- Health endpoint
- Versioned Chat API (`/api/v1/chat`)
- Pydantic request/response validation
- `ChatService`
- `LLMProvider` Protocol
- Fake LLM provider
- OpenAI LLM provider
- Environment-based configuration
- PDF document extraction
- Section-aware document chunking
- `EmbeddingProvider` Protocol, with fake and OpenAI implementations
- `VectorStore` Protocol, with in-memory and PostgreSQL+pgvector implementations
- Unit/integration tests

**Not implemented yet:**
- Document ingestion pipeline/endpoint (the stages above aren't wired together yet)
- Retrieval
- RAG
- Car-rental tools
- Agent/tool orchestration
- Conversation persistence
- Production deployment

## Project structure

```
app/
├── api/            # Router aggregation - versioning applied once, here
├── chats/           # Chat feature: router, schemas, ChatService
├── core/             # Cross-cutting config (Settings)
├── health/           # Health-check endpoint
├── knowledge/         # Document contracts, extraction, chunking, embedding, vector storage
└── llm/               # LLMProvider Protocol + fake/OpenAI implementations

tests/                # Mirrors the app/ layout, one test package per feature
data/
└── knowledge/        # Source PDFs for the future knowledge base
docker-compose.yml     # Local PostgreSQL + pgvector, for VECTOR_STORE_PROVIDER=pgvector
```

## Knowledge pipeline

**Current** (each stage implemented and tested; not yet wired together into one ingestion
flow — there is no ingestion endpoint or script that runs all of them in sequence):
```
PDF
  ↓
PdfDocumentExtractor
  ↓
ExtractedDocument
  ↓
SectionAwareChunker
  ↓
DocumentChunk[]
  ↓
EmbeddingProvider
  ↓
embedding vector[]
  ↓
VectorStore
  ↓
persisted, searchable chunks
```

**Planned:**
```
persisted, searchable chunks
  ↓
Retrieval
  ↓
LLM
```

The two PDFs in `data/knowledge/` (`car-rental-services.pdf`, `car-rental-policies.pdf`) are
the initial car-rental knowledge sources this pipeline will eventually be built around.

## LLM provider architecture

```
ChatService
  ↓
LLMProvider (Protocol)
  ↑
FakeLLMProvider / OpenAIProvider
```

`ChatService` depends only on the `LLMProvider` Protocol, never on a concrete provider or an
SDK. Provider-specific logic (the OpenAI SDK, its exceptions) is isolated inside
`app/llm/openai_provider.py` — nothing outside that one file imports the SDK.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

fastapi dev app/main.py
```

Available endpoints:
- `GET /health` — service health check
- `GET /docs` — interactive Swagger UI
- `POST /api/v1/chat` — chat endpoint (stub/echo reply unless an LLM provider is configured)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello"}'
```

## Configuration

Copy `.env.example` to `.env` and adjust as needed. Currently supported settings:

```
ENVIRONMENT=development
DEBUG=false

LLM_PROVIDER=fake        # "fake" or "openai"
OPENAI_API_KEY=sk-...     # required when LLM_PROVIDER=openai or EMBEDDING_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=fake             # "fake" or "openai" - independent of LLM_PROVIDER
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

VECTOR_STORE_PROVIDER=memory         # "memory" or "pgvector"
DATABASE_URL=postgresql+asyncpg://agentive:agentive@localhost:5432/agentive  # only for pgvector
```

`.env` is gitignored and must never be committed — only `.env.example`, with placeholder
values, is tracked.

## Testing

```bash
python -m pytest -v
```

Coverage currently spans the health and chat endpoints, request validation, the LLM provider
abstraction (fake, OpenAI with a mocked SDK client, and provider-selection/config-failure
behavior), the knowledge-document contracts, the PDF extractor (including an integration check
against the real PDFs in `data/knowledge/`), section-aware chunking (including chunk
statistics against the real PDFs), the embedding provider abstraction (fake, and OpenAI with a
mocked SDK client), and the vector store abstraction (in-memory unit tests plus
provider-selection/config-failure behavior). All of the above run without any external network
access, API key, or database.

A separate `tests/knowledge/test_pgvector_store_integration.py` exercises `PgVectorStore`
against a real PostgreSQL+pgvector instance; it's skipped automatically unless `DATABASE_URL`
is set, so it never affects the default `pytest -v` run. To run it locally:

```bash
docker compose up -d
DATABASE_URL=postgresql+asyncpg://agentive:agentive@localhost:5432/agentive \
    python -m pytest tests/knowledge/test_pgvector_store_integration.py -v
```

## PDF extraction

- [`pdfplumber`](https://github.com/jsvine/pdfplumber) is currently used for PDF text
  extraction, chosen after comparing it against `pypdf` on the actual project PDFs —
  `pdfplumber` reconstructs word/line spacing correctly out of the box, which matters for
  downstream RAG quality.
- PDF-library-specific code is isolated behind the `DocumentExtractor` Protocol
  (`app/knowledge/extraction.py`); nothing outside `app/knowledge/pdf_extractor.py` imports
  `pdfplumber`.
- The extractor preserves page-separated text and records `page_count` on the resulting
  `ExtractedDocument`.
- Extraction failures (corrupt files, missing files, no extractable text) are translated into
  the application's own `DocumentExtractionError` — no `pdfplumber`/`pdfminer` exception ever
  crosses that boundary.
- The two current PDFs (`data/knowledge/car-rental-services.pdf`,
  `data/knowledge/car-rental-policies.pdf`) have been successfully extracted and verified by
  the integration tests in `tests/knowledge/test_pdf_extractor_integration.py`.

## Embeddings

```
DocumentChunk[]
  ↓
EmbeddingProvider (Protocol)
  ↑
FakeEmbeddingProvider / OpenAIEmbeddingProvider
```

- `EmbeddingProvider` (`app/knowledge/embedding.py`) takes a batch of texts and returns one
  vector per text, in input order — ingestion will always embed many chunks at once, never
  one in isolation.
- `OpenAIEmbeddingProvider` uses OpenAI's `text-embedding-3-small` by default (a cost-efficient
  current-generation model); the openai SDK is isolated to
  `app/knowledge/openai_embedding_provider.py` alone. Output vector length is resolved from a
  small table of known models' dimensions rather than hard-coded, and construction fails
  clearly for an unrecognized model unless a dimension is passed explicitly.
- SDK/API failures are translated into the application's own `EmbeddingProviderError` —
  the same pattern `LLMProviderError` and `DocumentExtractionError` already use.
- `FakeEmbeddingProvider` returns deterministic, hash-derived vectors, so the default
  configuration and the full test suite never require an OpenAI API key or network access.
- Provider selection (`EMBEDDING_PROVIDER=fake|openai`) is independent of `LLM_PROVIDER` — a
  dedicated `get_embedding_provider()` composition function, not yet FastAPI-wired since no
  ingestion endpoint consumes it yet (consistent with `DocumentExtractor`/`DocumentChunker`
  having no `Depends()` wiring either).

## Vector storage

```
DocumentChunk + embedding
  ↓
VectorStore (Protocol)
  ↑
InMemoryVectorStore / PgVectorStore
```

**Technology: PostgreSQL + pgvector**, chosen over a dedicated vector database (Qdrant,
Pinecone, Weaviate) and over a bespoke lightweight store:

- **Metadata + vector search in one query.** RAG retrieval routinely needs "most similar
  chunks, optionally filtered by document/section" — pgvector answers that with a normal SQL
  query; a dedicated vector DB would need its metadata filtering feature to be reimplemented
  or duplicated.
- **One infrastructure dependency, not two, over this project's lifetime.** This service has
  no database today. The roadmap already commits to conversation persistence later, which
  will need a relational database regardless of what stores vectors. Choosing pgvector now
  means that one Postgres instance serves both needs; choosing a dedicated vector DB now would
  mean introducing Postgres separately later anyway.
- **Operational complexity matches actual scale.** Two PDFs, on the order of ten chunks
  today. A purpose-built ANN vector database earns its complexity at millions-of-vectors
  scale; nothing here is close to that, now or in any near-term plan.
- **Local dev and cost.** A single well-understood, widely-hosted (and often free-tier)
  database, versus standing up and paying for a second stateful service.

This is still "introducing another infrastructure dependency" — the project's own bar for
that (see Engineering principles) is cleared here because persistence is what `VectorStore`
inherently requires; it couldn't be built as a pure in-memory abstraction and still mean
anything.

**Storage model** — `VectorRecord` (write) and `VectorSearchResult` (read) wrap the existing
`DocumentChunk` rather than duplicating its fields or extending it: an embedding is a property
of how a chunk is *stored*, not a property of the chunk itself (the same chunk could be
re-embedded by a different model later), so `DocumentChunk` was left unchanged. `PgVectorStore`
persists `id, document_id, document_title, section_heading, text, position, embedding` -
exactly what a future retrieval/RAG layer needs to use and cite a result, nothing more.

**Similarity metric:** cosine similarity (pgvector's `<=>` operator; matched in pure Python for
`InMemoryVectorStore`), appropriate for OpenAI's embeddings, which are documented as
unit-normalized. Both implementations return a `score` where higher means more similar, so
retrieval code doesn't need to know which backend is active.

**Interface, not CRUD:** `VectorStore` exposes only `add` (batch upsert-by-chunk-id) and
`search` (top-k similarity) - not generic get/update/delete, because nothing in this codebase
needs to fetch or mutate one stored chunk in isolation yet.

**Local development:** `docker compose up -d` starts a `pgvector/pgvector:pg16` container
(see `docker-compose.yml`). `PgVectorStore.create_schema()` then creates the `vector`
extension and this store's table - it's an explicit call, not run automatically on app start.
The default `VECTOR_STORE_PROVIDER=memory` needs no database at all.

## Roadmap

- [x] FastAPI foundation
- [x] Chat API
- [x] LLM provider abstraction
- [x] OpenAI provider
- [x] PDF extraction foundation
- [x] Document chunking
- [x] Embedding provider
- [x] Vector store foundation
- [ ] Document ingestion pipeline
- [ ] Retrieval
- [ ] RAG
- [ ] Car-rental tools
- [ ] Agent orchestration
- [ ] Conversation/Q&A evaluation loop
- [ ] Production deployment

## Engineering principles

- **Thin API routes** — routers validate and delegate; no business logic in a router.
- **Dependency injection** — FastAPI's `Depends()` wires every layer together, with no
  manual container or global state.
- **Provider boundaries** — external systems (LLM SDKs, PDF libraries) sit behind a Protocol
  defined by the application layer, never imported by the code that consumes them.
- **Framework-independent knowledge layer** — `app/knowledge` has no FastAPI dependency,
  so it's usable (and testable) outside a request/response cycle.
- **Testability** — every provider boundary has a fake/mocked counterpart so the full test
  suite runs with no network access and no API keys.
- **Incremental implementation** — each layer is built with only as much abstraction as its
  current, real consumer justifies; speculative generality is avoided.
- **Avoiding unnecessary infrastructure** — no cache, queue, or search-specific database has
  been introduced speculatively. PostgreSQL (Step 9) is the first persistent store in this
  service, added only once `VectorStore` made persistence unavoidable, and chosen to also
  cover the roadmap's future relational needs (conversation persistence) instead of adding a
  second stateful dependency later.
