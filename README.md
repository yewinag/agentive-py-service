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
  ├── Chat              (app/chats)
  ├── RAG / Answering    (app/rag)
  ├── LLM Provider        (app/llm)
  └── Knowledge Pipeline   (app/knowledge)
```

Four boundaries, each with a single responsibility:

- **Chat/application layer** (`app/chats`) — orchestrates one HTTP request: validates input,
  delegates to `AnswerGenerator` via `ChatService`, shapes the response. `POST /api/v1/chat`
  now returns a knowledge-grounded answer (see Chat API integration section below).
- **RAG / grounded-answering layer** (`app/rag`) — combines retrieval and LLM generation into
  one grounded answer, behind `AnswerGenerator`. Depends on the `Retriever` and `LLMProvider`
  Protocols only. `ChatService` is its one real consumer.
- **LLM provider layer** (`app/llm`) — answers "how do we generate a reply?" behind an
  `LLMProvider` Protocol, so the concrete provider (a fake, OpenAI, or anything else later)
  is swappable without touching the chat or RAG layers.
- **Knowledge/document layer** (`app/knowledge`) — turns source documents into retrievable,
  embeddable, storable, and searchable units: extraction (`DocumentExtractor`), chunking
  (`DocumentChunker`), embedding (`EmbeddingProvider`), vector storage (`VectorStore`),
  semantic retrieval (`Retriever`), and ingestion orchestration (`IngestionService`, composing
  the previous four), each behind its own Protocol where one is warranted. Framework-independent
  itself (no FastAPI import); a thin startup hook in `app/main.py` is what actually calls it.

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
- `Retriever` Protocol, with a `VectorRetriever` implementation (semantic retrieval: query →
  embedding → vector search → ranked chunks)
- `IngestionService` (composes extraction → chunking → embedding → vector storage into one
  operation; not an HTTP endpoint)
- `AnswerGenerator` (RAG: retrieval → grounded prompt → `LLMProvider` → answer + sources)
- `POST /api/v1/chat` connected to the full RAG pipeline, returning a grounded reply and
  minimal source metadata
- Startup bootstrap of the two committed PDFs into the default in-memory knowledge store
- A clean, generic 503 response for embedding/vector-store/LLM failures (no raw SDK/database
  error ever reaches the client)
- Unit/integration tests

**Not implemented yet:**
- Conversation memory / chat history persistence
- Car-rental tools / function calling
- Agent orchestration / autonomous planning
- Q&A evaluation/improvement loop
- Streaming, reranking, hybrid search
- Authentication, business/rental database integration
- Production deployment

## Project structure

```
app/
├── api/            # Router aggregation + shared exception handlers
├── chats/           # Chat feature: router, schemas, ChatService (-> AnswerGenerator)
├── core/             # Cross-cutting config (Settings)
├── health/           # Health-check endpoint
├── knowledge/         # Document contracts, extraction, chunking, embedding, storage,
│                       # retrieval, ingestion orchestration, and startup bootstrap
├── llm/               # LLMProvider Protocol + fake/OpenAI implementations
└── rag/                # AnswerGenerator: Retriever + LLMProvider -> grounded answer

tests/                # Mirrors the app/ layout, one test package per feature
data/
└── knowledge/        # Source PDFs bootstrapped into the default knowledge store
docker-compose.yml     # Local PostgreSQL + pgvector, for VECTOR_STORE_PROVIDER=pgvector
```

## Knowledge pipeline

**Ingestion** (`IngestionService`, composing existing components - implemented and tested,
not exposed as an endpoint - see Ingestion section below):
```
PDF
  ↓
PdfDocumentExtractor  →  ExtractedDocument
  ↓
SectionAwareChunker    →  DocumentChunk[]
  ↓
EmbeddingProvider       →  embedding vector[]
  ↓
VectorStore              →  persisted, searchable chunks
```

**Retrieval + RAG, now reachable via `POST /api/v1/chat`:**
```
user message
  ↓
ChatService
  ↓
AnswerGenerator  →  Retriever  →  EmbeddingProvider  →  VectorStore.search()
  ↓                                      ↓
LLMProvider.generate_reply()   ranked VectorSearchResult[]
  ↓
ChatResponse { reply, sources[] }
```

Every stage is implemented, tested, and now wired end to end. Ingestion still has no HTTP
trigger - the default in-memory store is populated by an explicit startup bootstrap instead
(see Chat API integration below). Tools, agent orchestration, and conversation memory remain
future work.

The two PDFs in `data/knowledge/` (`car-rental-services.pdf`, `car-rental-policies.pdf`) are
the car-rental knowledge sources this pipeline is built, tested, and (by default) bootstrapped
around.

## LLM provider architecture

```
AnswerGenerator
  ↓
LLMProvider (Protocol)
  ↑
FakeLLMProvider / OpenAIProvider
```

As of Step 12, `ChatService` no longer talks to `LLMProvider` directly - it delegates to
`AnswerGenerator`, which is the layer that depends on the `LLMProvider` Protocol. Provider-specific
logic (the OpenAI SDK, its exceptions) stays isolated inside `app/llm/openai_provider.py` —
nothing outside that one file imports the SDK, and this didn't change when chat got wired to RAG.

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
- `POST /api/v1/chat` — returns a knowledge-grounded answer, by default drawn from the two
  PDFs bootstrapped into the in-memory store at startup (see Chat API integration below)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the minimum age to rent a car?"}'
```
```json
{
  "reply": "...",
  "sources": [
    {"document_title": "Terms & Rental Policies", "section_heading": "1. Driver Eligibility & Required Documents"}
  ]
}
```

With the default configuration (`LLM_PROVIDER=fake`, `EMBEDDING_PROVIDER=fake`), `reply` is a
`[fake-llm-reply]`-prefixed echo of the constructed prompt, not a real model answer - and
because `FakeEmbeddingProvider`'s vectors are hash-derived rather than semantically meaningful
(see Retrieval section), the retrieved sources for an unrelated question won't be properly
relevance-ranked either. Set `LLM_PROVIDER=openai` and `EMBEDDING_PROVIDER=openai` (with a real
`OPENAI_API_KEY`) for genuinely grounded, relevance-ranked answers.

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

RETRIEVAL_TOP_K=5                    # default VectorRetriever.retrieve() top_k
# RETRIEVAL_MIN_SCORE=0.75           # unset by default - see Retrieval section below
```

`.env` is gitignored and must never be committed — only `.env.example`, with placeholder
values, is tracked.

No new settings were needed for ingestion, RAG, or wiring chat to RAG: all of it reuses
`embedding_provider`, `vector_store_provider`, `llm_provider`, and
`retrieval_top_k`/`retrieval_min_score` as-is. The grounding system prompt is a fixed code
constant, not a setting - it's not something that should vary by environment, and making it
configurable would just be a place for inconsistent/untested prompt variants to creep in.

## Testing

```bash
python -m pytest -v
```

Coverage currently spans the health and chat endpoints, request validation, the LLM provider
abstraction (fake, OpenAI with a mocked SDK client, and provider-selection/config-failure
behavior), the knowledge-document contracts, the PDF extractor (including an integration check
against the real PDFs in `data/knowledge/`), section-aware chunking (including chunk
statistics against the real PDFs), the embedding provider abstraction (fake, and OpenAI with a
mocked SDK client), the vector store abstraction (in-memory unit tests plus
provider-selection/config-failure behavior), the retrieval layer (orchestration tests with
stub providers/stores, plus a deterministic end-to-end test using the real
`FakeEmbeddingProvider` + `InMemoryVectorStore`), `IngestionService` (fake-provider unit tests
plus an integration test processing the two real PDFs through the real extractor/chunker), and
`AnswerGenerator` (orchestration tests with stub `Retriever`/`LLMProvider`, plus a full
end-to-end test chaining `FakeDocumentExtractor → SectionAwareChunker → FakeEmbeddingProvider →
InMemoryVectorStore → Retriever → FakeLLMProvider`), the startup bootstrap (`bootstrap_default_knowledge_base`,
called directly - no FastAPI/lifespan involved), and the wired `/api/v1/chat` endpoint itself:
router-level tests (stubbed `ChatService`, proving the router only depends on
`get_chat_service()`'s abstraction), a full HTTP round-trip test chaining
`FakeDocumentExtractor → SectionAwareChunker → FakeEmbeddingProvider → InMemoryVectorStore →
Retriever → FakeLLMProvider → AnswerGenerator → ChatService →` the real FastAPI app, and a
parametrized test confirming embedding/vector-store/LLM failures all return a clean 503 with no
provider-specific detail in the body. All of the above run without any external network access,
API key, or database.

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

## Retrieval

```
query text
  ↓
Retriever (Protocol)
  ↑
VectorRetriever
  ├── EmbeddingProvider  (embeds the query)
  └── VectorStore        (finds the most similar stored chunks)
```

- `Retriever` (`app/knowledge/retriever.py`) depends only on the `EmbeddingProvider` and
  `VectorStore` Protocols — never a concrete OpenAI/pgvector type, and never FastAPI.
  `VectorRetriever` is pure orchestration: it embeds the query, calls `VectorStore.search()`,
  and returns the results. Query-embedding logic stays out of `VectorStore`; ranking/search
  logic stays out of `EmbeddingProvider`.
- **top_k** has one source of truth: `Settings.retrieval_top_k` (default `5`), read once by
  `get_retriever()` into `VectorRetriever`'s `default_top_k`, overridable per call
  (`retrieve(query, top_k=...)`). `VectorStore`'s own `top_k=5` default is never relied on -
  `VectorRetriever` always passes an explicit value. Results are never padded to reach top_k:
  if the store has fewer matches, fewer are returned, exactly as `VectorStore.search()` says.
- **Similarity threshold (`min_score`)** exists as a parameter and a `Settings.retrieval_min_score`
  setting but is **disabled by default** (`None`). There is no empirical basis yet for a
  universal cosine-similarity cutoff for this embedding model/knowledge base — no labeled
  queries, no evaluation loop (that's the roadmap's still-future "Conversation/Q&A evaluation
  loop"), and the default `FakeEmbeddingProvider`'s hash-derived vectors aren't semantically
  meaningful anyway, so there's nothing to calibrate a default against honestly. The filtering
  itself is implemented and tested; a real value can be set later via `RETRIEVAL_MIN_SCORE`
  once there's data to justify one, with no code change.
- Errors are not re-wrapped: `EmbeddingProviderError`/`VectorStoreError` propagate through
  `Retriever` unchanged — it calls no raw SDK itself, only Protocols that already translate
  their own failures.
- `get_retriever(settings)` is a plain function, not FastAPI `Depends()`-wired, for the same
  reason as `get_embedding_provider()`/`get_vector_store()`: there is still no retrieval
  endpoint. It composes its own `EmbeddingProvider`/`VectorStore` via their existing
  composition points rather than deciding providers itself.

## Ingestion

`IngestionService` (`app/knowledge/ingestion.py`) composes the four existing knowledge Protocols
into one operation - `DocumentExtractor → DocumentChunker → EmbeddingProvider → VectorStore` -
reimplementing none of them:

```python
async def ingest(self, sources: list[DocumentSource]) -> int: ...
```

- Extraction and chunking are **not** `Settings`-selected the way the LLM/embedding/vector-store
  providers are: there's only one real implementation of each worth choosing for actual
  ingestion (`PdfDocumentExtractor`, `SectionAwareChunker`) - the `Fake*` variants exist purely
  for tests, never as a real ingestion option, so there's nothing to select between.
- Deliberately **not an HTTP endpoint**: ingestion is an occasional, batch operation (run when
  the knowledge base changes), not a per-request one. No file upload API, admin CRUD, or
  scheduling - those are separate future concerns, out of scope for this foundation.
- `add()` on `VectorStore` is upsert-by-chunk-id, so re-running ingestion on the same sources
  doesn't duplicate chunks - verified by `tests/knowledge/test_ingestion_integration.py`, which
  ingests the two real PDFs twice and confirms the stored count doesn't grow.
- `get_ingestion_service(settings)` is the composition point, following the same
  plain-function, no-`Depends()` pattern as every other `app/knowledge` composition function.

## RAG / grounded answering

```
Retriever  →  VectorSearchResult[]
                    ↓
AnswerGenerator (Protocol-free, like ChatService)
  ├── build_context()  →  deterministic text block: document title + section heading + chunk text
  ├── build_prompt()    →  grounding instructions + context + question
  └── LLMProvider.generate_reply(prompt)  →  answer
                    ↓
GroundedAnswer { answer, sources[] }
```

- **`AnswerGenerator`** (`app/rag/answer_generator.py`) depends only on the `Retriever` and
  `LLMProvider` Protocols - never OpenAI, pgvector, or FastAPI directly. It has no Protocol of
  its own: like `ChatService`, it's application orchestration logic with exactly one real
  implementation, not an external boundary with swappable backends - testability already comes
  from `Retriever`/`LLMProvider` each being fakeable.
- **Context construction** (`build_context`) is a small, pure, directly-tested function. It
  includes only `document_title`, `section_heading`, and `text` per retrieved chunk - not
  `score`/`id`/`position`, which mean nothing to the model and would just be prompt noise.
  Same input always produces the same string.
- **Grounding**: a fixed, concise system instruction (not configuration - see Configuration
  below for why) tells the model to answer only from the supplied context, never invent
  policies/prices/requirements/availability, and say so explicitly when the context is
  insufficient. `LLMProvider.generate_reply()` still takes one plain string, unchanged from
  Step 3/4 - the instructions, context, and question are composed into that single string by
  `build_prompt()`, so the existing `LLMProvider` Protocol needed no changes.
- **Empty retrieval**: if `Retriever.retrieve()` returns no results, `AnswerGenerator` returns
  a fixed `NOT_AVAILABLE_ANSWER` and **never calls the LLM** - tested explicitly. The
  application layer decides "we have nothing relevant," not the model.
- **Similarity threshold**: unchanged from Step 10 - `AnswerGenerator` does not add a second
  threshold. `Retriever`'s `min_score` (still off by default) remains the only place that
  decides which results are relevant enough to use.
- **`ChatService` integration**: as of Step 12, `ChatService` delegates to `AnswerGenerator`
  (see Chat API integration below) rather than folding retrieval into `ChatService` directly -
  `ChatService` would otherwise duplicate orchestration `AnswerGenerator` already owns, and the
  provider-agnostic `LLMProvider` boundary stays cleanest when exactly one thing composes it
  for grounded answers (`AnswerGenerator`) and `ChatService` just calls that.

## Chat API integration

```
POST /api/v1/chat
  ↓
ChatRequest (validated)
  ↓
ChatService.get_reply()
  ↓
AnswerGenerator.answer()  →  GroundedAnswer { answer, sources[] }
  ↓
ChatResponse { reply, sources[] }
```

- **`ChatService`** (`app/chats/service.py`) now holds an `AnswerGenerator`, not an
  `LLMProvider` - it knows WHAT it needs (an answer to a message), never that retrieval,
  embeddings, vector search, or an LLM SDK are involved. `get_chat_service()` is the one
  FastAPI-`Depends()`-wired seam in this whole chain: it resolves `Settings` via
  `Depends(get_settings)`, then calls the plain `get_answer_generator(settings)` composition
  function - the same bridge pattern every other layer's composition function already
  documented as its own eventual FastAPI entry point.
- **Response contract**: `ChatResponse` gained `sources: list[AnswerSource]` alongside the
  existing `reply: str` - reusing `AnswerSource` from `app/rag/models.py` directly rather than
  redefining an identical schema. Only `document_title` and `section_heading` are exposed;
  chunk text, ids, positions, and similarity scores are not - a user can see *where* an answer
  came from without the response leaking internal retrieval detail. `ChatRequest` is unchanged.
- **Runtime knowledge availability**: `InMemoryVectorStore` (the default) is empty at the start
  of every process, and - a real bug caught while building this step - `get_vector_store()`
  used to construct a *fresh* instance on every call, so even a populated store would never
  have been visible to a later request. Fixed by making `get_vector_store()` return a
  process-wide singleton for `vector_store_provider=memory` (a `PgVectorStore` needs no such
  fix - its data already lives in an external, already-shared database). On top of that fix, an
  explicit `lifespan` hook in `app/main.py` calls `bootstrap_default_knowledge_base()`
  (`app/knowledge/bootstrap.py`) once at startup, which reuses `IngestionService` to ingest the
  two committed PDFs - no ingestion logic is duplicated. It's a no-op for `pgvector` (a real,
  persistent, already-shared database is ingested deliberately, not silently on every boot -
  see the Ingestion section), idempotent (`VectorStore.add()` is upsert-by-chunk-id), and
  testable in isolation (called directly in tests, with no FastAPI/lifespan involved).
  Bootstrap never runs per-request - only once, at process startup.
- **Error handling**: `EmbeddingProviderError`, `VectorStoreError`, and `LLMProviderError` -
  already-translated application exceptions, never raw SDK/database errors - are mapped by a
  shared handler (`app/api/exception_handlers.py`) to one generic `503 {"detail": "..."}`
  response, registered once in `app/main.py`. The handler never echoes the exception's own
  message, so a connection string, API-key hint, or provider name can't leak to the client.
  Invalid requests still return `422` (unchanged, Pydantic-validated) and an empty/irrelevant
  retrieval result still returns `200` with the deterministic "not available" reply (unchanged
  from Step 11) - neither is treated as an error.

## Roadmap

- [x] FastAPI foundation
- [x] Chat API
- [x] LLM provider abstraction
- [x] OpenAI provider
- [x] PDF extraction foundation
- [x] Document chunking
- [x] Embedding provider
- [x] Vector store foundation
- [x] Semantic retrieval
- [x] Ingestion orchestration (`IngestionService`, not an endpoint)
- [x] RAG / grounded answering (`AnswerGenerator`)
- [x] Chat API integration (`POST /api/v1/chat` connected to the full RAG pipeline)
- [ ] Conversation memory / chat history persistence
- [ ] Car-rental tools / function calling
- [ ] Agent orchestration / autonomous planning
- [ ] Q&A evaluation/improvement loop
- [ ] Streaming, reranking, hybrid search
- [ ] Authentication, business/rental database integration
- [ ] Production deployment

## Engineering principles

- **Thin API routes** — routers validate and delegate; no business logic in a router.
- **Dependency injection** — FastAPI's `Depends()` wires every layer together, with no
  manual container or global state.
- **Provider boundaries** — external systems (LLM SDKs, PDF libraries) sit behind a Protocol
  defined by the application layer, never imported by the code that consumes them.
- **Framework-independent knowledge and RAG layers** — neither `app/knowledge` nor `app/rag`
  has a FastAPI dependency, so both are usable (and testable) outside a request/response cycle.
- **Testability** — every provider boundary has a fake/mocked counterpart so the full test
  suite runs with no network access and no API keys.
- **No leaking internal failures to the client** — provider/storage exceptions are already
  translated once (Steps 4/8/9); Step 12 adds one shared handler mapping them to a generic
  503, so a connection string or API-key hint can never surface in an HTTP response body.
- **Incremental implementation** — each layer is built with only as much abstraction as its
  current, real consumer justifies; speculative generality is avoided.
- **Avoiding unnecessary infrastructure** — no cache, queue, or search-specific database has
  been introduced speculatively. PostgreSQL (Step 9) is the first persistent store in this
  service, added only once `VectorStore` made persistence unavoidable, and chosen to also
  cover the roadmap's future relational needs (conversation persistence) instead of adding a
  second stateful dependency later.
