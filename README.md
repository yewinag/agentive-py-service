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
  ├── Chat                          (app/chats)
  ├── Agent / Orchestration           (app/agent)
  ├── Conversation Memory               (app/conversation)
  ├── RAG / Answering                     (app/rag)
  ├── Tools                                 (app/tools)  →  (future) NestJS Business API
  ├── LLM Provider                             (app/llm)
  └── Knowledge Pipeline                          (app/knowledge)
```

Seven boundaries, each with a single responsibility:

- **Chat/application layer** (`app/chats`) — orchestrates one HTTP request: validates input,
  resolves/continues a conversation, delegates to `AgentService` via `ChatService`, shapes the
  response. `POST /api/v1/chat` returns a knowledge-grounded, tool-assisted, multi-turn-aware
  answer - the caller never sees whether a tool was involved.
- **Agent/orchestration layer** (`app/agent`) — the new capability this step adds: offers
  available tools to the LLM alongside retrieved knowledge, executes at most one bounded round
  of tool calls it requests, and returns a final grounded answer, behind `AgentService`. Depends
  on `Retriever`, `LLMProvider`, and `ToolRegistry` - reuses `AnswerGenerator`'s prompt-building
  helpers rather than duplicating them (see Agent / tool calling section below).
- **Conversation memory layer** (`app/conversation`) — a domain deliberately separate from the
  knowledge base and from tools (see Conversation context section below): identity, ordering,
  and bounded recent-history for a conversation, behind `ConversationStore`. Holds no car-rental
  knowledge, no embeddings, no vectors, no tool-call structures.
- **RAG / grounded-answering layer** (`app/rag`) — combines retrieval, bounded conversation
  context, and LLM generation into one grounded answer, behind `AnswerGenerator`. Depends on
  the `Retriever` and `LLMProvider` Protocols only. Still fully working and tested; `AgentService`
  is now `ChatService`'s actual dependency (a strict superset of this capability) - see Agent /
  tool calling section for why `AnswerGenerator` wasn't modified or removed.
- **Tools layer** (`app/tools`) — a domain deliberately separate from both the knowledge base
  and the business backend itself (see Tools section below): describes and executes dynamic
  business operations (today: vehicle availability) behind a `Tool` Protocol and a
  `ToolRegistry`, talking to the business system through a `BusinessServiceClient` Protocol.
  Owns no rental inventory, booking state, or customer data - it's an integration boundary, not
  a business-domain owner. `AgentService` is its first real consumer.
- **LLM provider layer** (`app/llm`) — answers "how do we generate a reply, optionally offering
  tools?" behind an `LLMProvider` Protocol, so the concrete provider (a fake, OpenAI, or
  anything else later) is swappable without touching the chat, agent, conversation, RAG, or
  tools layers.
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
- `ConversationStore` Protocol, with an in-memory implementation (`Conversation`,
  `ConversationMessage`: identity, ordering, bounded recent-history - kept a separate domain
  from the knowledge base, never stored in the vector store)
- Multi-turn conversations over `POST /api/v1/chat`: an optional `conversation_id` in the
  request, echoed in the response, carrying a bounded recent-message window into the prompt
- `Tool` Protocol, `ToolMetadata`, and `ToolRegistry` (register/resolve/list - no dynamic
  plugin loading)
- `BusinessServiceClient` Protocol, with a fake/deterministic implementation
- One concrete tool: `check_vehicle_availability` (dynamic availability lookup, validated
  input, distinguishing input/business-service/execution failures)
- Provider-agnostic tool calling: `LLMProvider.generate()` accepts tools and returns either
  final text or a structured tool-call request, expressed entirely in application-level models
  (`LLMRequest`, `LLMResponse`, `LLMMessage`, `ToolCall`) - no OpenAI SDK type ever leaves
  `app/llm/openai_provider.py`
- `AgentService`: offers `check_vehicle_availability` to the LLM, executes it if requested
  (bounded to exactly one round), and returns a final grounded answer - `ChatService`'s actual
  dependency for `POST /api/v1/chat` as of this step
- Unit/integration tests

**Not implemented yet:**
- Conversation persistence beyond a single process (no PostgreSQL-backed `ConversationStore`
  yet - see Conversation context section for why)
- Long-term/semantic conversation memory, summarization, or query rewriting for follow-ups
- Real NestJS Business API integration (no HTTP `BusinessServiceClient` yet - see Tools section)
- More than one tool-call round, parallel tool execution, or multiple registered tools beyond
  `check_vehicle_availability`
- Autonomous planning / multi-agent systems
- Booking creation, modification, or cancellation of any kind
- Q&A evaluation/improvement loop
- Streaming, reranking, hybrid search
- Authentication, business/rental database integration
- Production deployment

## Project structure

```
app/
├── agent/            # AgentService: Retriever + LLMProvider + ToolRegistry -> grounded answer
├── api/                # Router aggregation + shared exception handlers
├── chats/               # Chat feature: router, schemas, ChatService (-> AgentService + ConversationStore)
├── conversation/         # Conversation, ConversationMessage, ConversationStore Protocol + in-memory impl
├── core/                 # Cross-cutting config (Settings)
├── health/               # Health-check endpoint
├── knowledge/            # Document contracts, extraction, chunking, embedding, storage,
│                         # retrieval, ingestion orchestration, and startup bootstrap
├── langchain_integration/ # Sole isolation boundary for the langchain/langchain-openai/
│                         # langchain-qdrant SDKs (Phase 2.7.1: dependency only, no chain yet)
├── llm/                  # LLMProvider Protocol, provider-agnostic models, fake/OpenAI implementations
├── rag/                  # AnswerGenerator: Retriever + LLMProvider + history -> grounded answer (no tools)
└── tools/                # Tool/ToolRegistry, BusinessServiceClient, check_vehicle_availability

tests/                # Mirrors the app/ layout, one test package per feature
data/
└── knowledge/        # Source PDFs bootstrapped into the default knowledge store
docker-compose.yml     # Local PostgreSQL+pgvector (VECTOR_STORE_PROVIDER=pgvector) and Qdrant
                       # (VECTOR_STORE_PROVIDER=qdrant) - the AI knowledge store for this
                       # project; unrelated to the separate Strapi/PostgreSQL business database
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

**Retrieval + RAG + tools + conversation context, reachable via `POST /api/v1/chat`:**
```
user message + optional conversation_id
  ↓
ChatService  →  ConversationStore  (resolve conversation, load bounded recent history)
  ↓
AgentService  →  Retriever  →  EmbeddingProvider  →  VectorStore.search()
  ↓                                      ↓
LLMProvider.generate()          ranked VectorSearchResult[]
  ├── final text  ──────────────────────────────────────────────┐
  └── tool call → ToolRegistry → Tool → BusinessServiceClient    │
        → result fed back → LLMProvider.generate() (no tools) ───┘
  ↓
ChatService  →  ConversationStore  (append user + assistant messages)
  ↓
ChatResponse { reply, sources[], conversation_id }
```

Every stage is implemented, tested, and wired end to end. Ingestion still has no HTTP
trigger - the default in-memory store is populated by an explicit startup bootstrap instead
(see Chat API integration below). Real NestJS integration, autonomous/multi-round agent
planning, conversation summarization, and durable (cross-process) conversation persistence
remain future work.

The six PDFs in `data/knowledge/` (`01-rental-services.pdf`, `02-rental-policies.pdf`,
`03-booking-policy.pdf`, `04-cancellation-policy.pdf`, `05-payment-policy.pdf`,
`06-pickup-return-policy.pdf`) are the canonical car-rental knowledge sources this pipeline is
built, tested, and (by default) bootstrapped around. Two earlier, broader PDFs
(`car-rental-services.pdf`, `car-rental-policies.pdf`) covered similar ground with some
conflicting numbers (e.g. cancellation fee percentages, deposit refund windows) and have been
retired in favor of these six more focused, mutually-consistent documents.

## LLM provider architecture

```
AnswerGenerator (no tools)   AgentService (offers tools)
          ↓                           ↓
          └──────────→ LLMProvider (Protocol) ←──────────┘
                              ↑
                  FakeLLMProvider / OpenAIProvider
```

`ChatService` doesn't talk to `LLMProvider` directly (since Step 12) - it delegates to
`AgentService` (since Step 15), which is one of two layers that now depend on the `LLMProvider`
Protocol; `AnswerGenerator` still depends on it too, unchanged. Provider-specific logic (the
OpenAI SDK, its exceptions, and - new in Step 15 - all translation to/from its tool-calling
format) stays isolated inside `app/llm/openai_provider.py` — nothing outside that one file
imports the SDK. See Agent / tool calling below for how `LLMProvider` itself evolved this step.

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
- `POST /api/v1/chat` — returns a knowledge-grounded, tool-assisted, multi-turn-aware answer, by
  default drawn from the six canonical PDFs bootstrapped into the in-memory store at startup (see
  Chat API integration, Conversation context, and Agent / tool calling below). The response never
  reveals whether a tool was used - the contract stays `reply`/`sources`/`conversation_id`
  either way.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the minimum age to rent a car?"}'
```
```json
{
  "reply": "...",
  "sources": [
    {"document_title": "Rental Policies", "section_heading": "1. Eligibility Requirements"}
  ],
  "conversation_id": "6a0fa43e630b4c418d63e9735dbfe4ea"
}
```

Continue the same conversation by passing `conversation_id` back on the next request:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"And what documents do I need?","conversation_id":"6a0fa43e630b4c418d63e9735dbfe4ea"}'
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

VECTOR_STORE_PROVIDER=memory         # "memory", "pgvector", or "qdrant"
DATABASE_URL=postgresql+asyncpg://agentive:agentive@localhost:5432/agentive  # only for pgvector
QDRANT_URL=http://localhost:6333             # only for qdrant
QDRANT_COLLECTION=knowledge_chunk_embeddings # only for qdrant

RETRIEVAL_TOP_K=5                    # default VectorRetriever.retrieve() top_k
# RETRIEVAL_MIN_SCORE=0.75           # unset by default - see Retrieval section below

CONVERSATION_STORE_PROVIDER=memory   # only "memory" is implemented today
CONVERSATION_HISTORY_WINDOW=6        # last N messages (~3 turns) sent to the LLM as context

BUSINESS_SERVICE_PROVIDER=fake       # only "fake" is implemented today - see Tools section
```

`.env` is gitignored and must never be committed — only `.env.example`, with placeholder
values, is tracked.

Ingestion and wiring chat to RAG needed no new settings. Step 13 added
`conversation_store_provider`/`conversation_history_window`; Step 14 added
`business_service_provider`. Step 15 adds **no new settings at all**: `AgentService` is built
from the same `get_retriever()`/`get_llm_provider()`/`get_tool_registry()` composition
functions every other layer already uses. The one-tool-call-round bound
(`MAX_TOOL_ROUNDS` in `app/agent/service.py`) is deliberately a code constant, not a setting -
unlike `retrieval_top_k` or `conversation_history_window` (genuine operational tuning knobs,
safe to default without usage data), this bound is a scoped safety constraint for this step
(see Scope restrictions), and making it configurable would let it be silently raised past what
this step was actually built and tested for. The grounding system prompt remains a fixed code
constant too, for the same reason as always - not something that should vary by environment.

Phase 2.5 (Qdrant) adds `QDRANT_URL`/`QDRANT_COLLECTION` only. There is deliberately no separate
`QDRANT_VECTOR_SIZE` setting: vector width is derived from whichever embedding provider is
actually configured (`get_vector_store()`'s `_embedding_dimensions_for()`, shared by `pgvector`
and `qdrant`), so the two can never silently drift apart.

**Phase 2.6 fix:** `_embedding_dimensions_for()` used to resolve dimensions from
`OPENAI_EMBEDDING_MODEL` directly, regardless of `EMBEDDING_PROVIDER` - so `VECTOR_STORE_PROVIDER=
qdrant`/`pgvector` combined with the default `EMBEDDING_PROVIDER=fake` would size the
collection/table for an OpenAI model's dimensions instead of `FakeEmbeddingProvider`'s, and writes
would fail with a dimension mismatch. It now instead builds the real, configured
`EmbeddingProvider` (`get_embedding_provider(settings)`) and reads its `dimensions` property - the
same instance that will actually do the embedding - so the vector store is always sized correctly
for whichever provider (`fake` or `openai`) is selected, without a second, independently
configured dimension value to keep in sync. Both `PgVectorStore` and `QdrantVectorStore` expose
their resolved width via a `.dimensions` property, so this agreement is directly testable rather
than only inferred from a successful write (`tests/knowledge/test_vector_store_selection.py`).

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
plus an integration test processing all six real, canonical PDFs through the real
extractor/chunker), and
`AnswerGenerator` (still fully covered exactly as in Step 11/13 - a stub `LLMProvider` and
directly-tested `build_context`/`build_history_block`/`build_prompt`, plus a full end-to-end
test chaining `FakeDocumentExtractor → SectionAwareChunker → FakeEmbeddingProvider →
InMemoryVectorStore → Retriever → FakeLLMProvider`), the startup bootstrap
(`bootstrap_default_knowledge_base`, called directly - no FastAPI/lifespan involved), and the
wired `/api/v1/chat` endpoint itself: router-level tests (stubbed `ChatService`, proving the
router only depends on `get_chat_service()`'s abstraction), a full HTTP round-trip test
chaining fakes through `AgentService → InMemoryConversationStore → ChatService →` the real
FastAPI app - including a two-request test proving a follow-up's prompt genuinely contains the
prior turn's exact question - and a parametrized test confirming embedding/vector-store/LLM
failures all return a clean 503 with no provider-specific detail in the body. `ConversationStore`
has its own dedicated coverage: creating a conversation, appending messages in order, a bounded
recent-message window, a missing conversation id (`get()` returns `None`;
`append_message`/`get_recent_messages` raise `ConversationNotFoundError`), and isolation between
two unrelated conversations. The tool boundary has its own coverage too: `ToolRegistry`
(register, resolve, list, unknown-name lookup, duplicate-registration rejection), the metadata/
input-schema/valid-execution/invalid-input/error-propagation contract for
`CheckVehicleAvailabilityTool`, `FakeBusinessServiceClient`'s deterministic date-range/category
filtering, and provider-selection composition tests.

Step 15 (provider-agnostic tool calling) added: `LLMMessage`/`LLMRequest`/`LLMResponse`/
`ToolCall` model tests; `FakeLLMProvider`'s two modes (default single-call echo, and a
configured response sequence for scripting tool-call-then-final-answer flows); `OpenAIProvider`
tests with a mocked SDK client verifying tool-metadata-to-OpenAI-format conversion, tool-call
parsing from the response, correct re-translation of a prior assistant tool-call + tool-result
turn for the follow-up request, and malformed-arguments handling; and `AgentService`'s own
orchestration tests - final answer returned directly, a tool call executed and its result fed
back to the model, tools omitted (not just declined) on the bounded follow-up call, unknown
tool/invalid arguments/business-service failure/tool execution failure each handled gracefully
without aborting the request, LLM failures on either call propagating, and the one-round policy
enforced even against a misbehaving provider that ignores the empty `tools` list. Two further
integration tests chain the real `CheckVehicleAvailabilityTool` + `FakeBusinessServiceClient`
through `AgentService`, and one chains the whole stack through the real FastAPI app via
`POST /api/v1/chat`, proving a tool-assisted answer end to end with the public response
contract (`reply`/`sources`/`conversation_id`) unchanged. All of the above run without any
external network access, API key, or database.

A separate `tests/knowledge/test_pgvector_store_integration.py` exercises `PgVectorStore`
against a real PostgreSQL+pgvector instance; it's skipped automatically unless `DATABASE_URL`
is set, so it never affects the default `pytest -v` run. To run it locally:

```bash
docker compose up -d
DATABASE_URL=postgresql+asyncpg://agentive:agentive@localhost:5432/agentive \
    python -m pytest tests/knowledge/test_pgvector_store_integration.py -v
```

Similarly, `tests/knowledge/test_qdrant_vector_store_integration.py` exercises `QdrantVectorStore`
against a real Qdrant instance; it's skipped automatically unless `QDRANT_URL` is set. Covers
collection creation (and idempotent re-creation), upsert, search, metadata preservation, an
empty/never-created collection returning `[]` rather than erroring, multiple documents/chunks,
and - via a second `QdrantVectorStore` built from a fresh client against the same collection name
- that data survives what amounts to an application restart. To run it locally:

```bash
docker compose up -d qdrant
QDRANT_URL=http://localhost:6333 python -m pytest tests/knowledge/test_qdrant_vector_store_integration.py -v
```

`tests/knowledge/test_ingest_command.py` covers the explicit ingestion command's
(`app/knowledge/ingest.py`) own control flow without any real Qdrant or PDFs (it refuses cleanly,
exit code 1, for every non-`qdrant` `VECTOR_STORE_PROVIDER`) - it never needs `QDRANT_URL` and
always runs. `tests/knowledge/test_qdrant_ingestion_command_integration.py` is the real,
skip-gated (`QDRANT_URL`) round trip: all six canonical PDFs discovered, 22 chunks/vectors
produced, a second run staying at 22 (idempotent), `--reset` recreating the collection without
duplicating, and a fresh `QdrantVectorStore` retrieving what a prior run persisted.
`tests/knowledge/test_bootstrap.py` also gained a `discover_canonical_sources()` unit test and a
`vector_store_provider="qdrant"` no-op case, alongside its existing `pgvector` one.

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
- The six canonical PDFs in `data/knowledge/` (`01-rental-services.pdf` through
  `06-pickup-return-policy.pdf`) have been successfully extracted and verified by the
  integration tests in `tests/knowledge/test_pdf_extractor_integration.py`.

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
InMemoryVectorStore / PgVectorStore / QdrantVectorStore
```

**Qdrant is this project's AI knowledge store; PostgreSQL is not.** The wider system
architecture settled on a clear split: PostgreSQL (via the separate Strapi project) owns
business/transactional data (Cars, Bookings, Payments) and is never touched by this service;
Qdrant owns AI knowledge (the six canonical PDFs, as embeddings) for this service. `PgVectorStore`
remains implemented and tested below - it predates that split and is kept as a working
`VectorStore` implementation - but `qdrant` is the intended `VECTOR_STORE_PROVIDER` for this
project going forward. This project's own `docker-compose.yml` also runs a `postgres` container
for `PgVectorStore`, but that is a private, disposable local database for this optional vector
store only - it has no relationship to, and shares no data with, the Strapi business database.

**Why not pgvector after all:** the reasoning below (metadata + vector search in one query, one
infrastructure dependency) was sound in isolation, but assumed this service would eventually own
a relational database for its own needs (e.g. conversation persistence). The confirmed system
architecture instead keeps this service's only stateful dependency AI-knowledge-shaped
(Qdrant) and leaves relational/business persistence entirely to the separate Strapi/PostgreSQL
project - so the "one Postgres serves both needs" argument no longer applies.

**Technology comparison, updated:**

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

**Similarity metric:** cosine similarity (pgvector's `<=>` operator, converted from distance to
similarity; Qdrant's `Distance.COSINE`, already a similarity so no conversion needed; matched in
pure Python for `InMemoryVectorStore`) - appropriate for OpenAI's embeddings, which are
documented as unit-normalized. All three implementations return a `score` where higher means
more similar, so retrieval code doesn't need to know which backend is active.

**Interface, not CRUD:** `VectorStore` exposes only `add` (batch upsert-by-chunk-id) and
`search` (top-k similarity) - not generic get/update/delete, because nothing in this codebase
needs to fetch or mutate one stored chunk in isolation yet.

**`QdrantVectorStore`** (`app/knowledge/qdrant_vector_store.py`) stores each chunk as a Qdrant
point, with the chunk's fields (`chunk_id, document_id, document_title, section_heading, text,
position`) in the point's payload - the same information `PgVectorStore` keeps in table columns,
just payload-shaped instead of row-shaped. One real constraint the abstraction had to absorb:
Qdrant point ids must be an unsigned integer or a UUID, but this project's chunk ids are strings
like `01-rental-services.pdf-chunk-0`. `QdrantVectorStore` derives a deterministic UUID from each
chunk id (`uuid.uuid5` against a fixed namespace) to use as the actual point id, and keeps the
real chunk id in the payload; `add()` therefore still upserts by chunk id exactly as the
`VectorStore` contract requires, and every caller only ever sees the original string chunk id.
`ensure_collection()` (create-if-missing) and `delete_collection()` mirror `PgVectorStore`'s
`create_schema()`/`drop_schema()` - explicit, not run automatically on app start.

**Local development:** `docker compose up -d` starts both a `pgvector/pgvector:pg16` container
(for `PgVectorStore`) and a `qdrant/qdrant:latest` container (for `QdrantVectorStore`), each with
its own named, persistent volume - so vectors survive a container restart. `PgVectorStore`'s
`create_schema()` / `QdrantVectorStore`'s `ensure_collection()` are both explicit calls, not run
automatically on app start. The default `VECTOR_STORE_PROVIDER=memory` needs neither database.

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
  ingests all six real, canonical PDFs twice and confirms the stored count stays at 22.
- `get_ingestion_service(settings)` is the composition point, following the same
  plain-function, no-`Depends()` pattern as every other `app/knowledge` composition function.

### Explicit knowledge ingestion command (Qdrant)

**FastAPI startup ≠ knowledge ingestion.** `bootstrap_default_knowledge_base()` (see Chat API
integration) only ever populates the transient, in-process `InMemoryVectorStore` - it is a no-op
for every persistent store (`pgvector`, `qdrant`), on purpose: a persistent store already has
whatever was ingested into it, and re-running real embedding calls on every app boot for no
benefit would be wasteful. Getting the six canonical PDFs into Qdrant is therefore a deliberate,
standalone, explicit operation - not something starting the FastAPI app ever does for you:

```bash
docker compose up -d qdrant
VECTOR_STORE_PROVIDER=qdrant python -m app.knowledge.ingest
```

```
Knowledge ingestion started
Documents discovered: 6
Documents processed: 6
Chunks generated: 22
Vectors upserted: 22
Collection: knowledge_chunk_embeddings
Status: SUCCESS
```

This command (`app/knowledge/ingest.py`) needs no running FastAPI process - it composes the same
`PdfDocumentExtractor`/`SectionAwareChunker`/`EmbeddingProvider`/`VectorStore` abstractions
`IngestionService` already orchestrates, wired via the same `Settings`/composition-function
pattern every other layer uses (`get_embedding_provider(settings)`, `get_vector_store(settings)`)
- no ingestion logic was duplicated to build it. It refuses to run (exit code 1, no PDFs touched)
unless `VECTOR_STORE_PROVIDER=qdrant`, since it exists specifically to persist knowledge into
Qdrant.

**Idempotent by construction:** running it twice does not duplicate vectors. Each `DocumentChunk`
has a stable id (`{document_id}-chunk-{n}`); `QdrantVectorStore` derives a deterministic point id
from that chunk id (see Vector storage), so `add()` always upserts the same point rather than
creating a new one. First run: 6 PDFs → 22 chunks → 22 points. Second run: 6 PDFs → 22 chunks →
still 22 points, each one's content refreshed in place.

**Persistence:** Qdrant's data lives in the `agentive_qdrant_data` docker volume (see
`docker-compose.yml`), not in the Python process. A fresh `QdrantVectorStore`/client - including
across a full container restart (`docker compose restart qdrant`), not just a new app process -
sees everything a prior ingestion run wrote; verified live for this phase, not just asserted.

**Handling stale documents:** the ingestion command only ever adds/updates chunks for PDFs
currently in `data/knowledge/` - it never deletes a point for a document that used to exist but
doesn't anymore. Renaming or removing a canonical PDF therefore leaves its old chunks orphaned in
Qdrant (retrievable forever, even though the source document is gone) unless you pass `--reset`:

```bash
VECTOR_STORE_PROVIDER=qdrant python -m app.knowledge.ingest --reset
```

`--reset` deletes and recreates the collection (`QdrantVectorStore.reset_collection()`) before
ingesting, so the result always exactly matches whatever is currently in `data/knowledge/` - a
full rebuild, not an incremental sync. This is a deliberately simple tradeoff for six PDFs that
change rarely: no document-versioning, no diffing of "which documents changed since last time",
just "empty, then fully re-ingest" when you know the canonical set changed. That full rebuild
re-embeds every chunk (cheap today with `FakeEmbeddingProvider`; a real cost once Phase 2.7 wires
up OpenAI embeddings) - a fine cost for six small documents, but the reason this stays a manual
flag rather than something the command decides to do automatically.

### Using real OpenAI embeddings for production ingestion (Phase 2.7.2)

`EMBEDDING_PROVIDER` selects between `fake` and `openai` completely independently of
`VECTOR_STORE_PROVIDER` - nothing about Qdrant is special-cased in `app/knowledge/ingest.py`,
which only ever calls the generic `get_embedding_provider(settings)` composition point, exactly
like every other consumer. To persist real, semantically meaningful embeddings into Qdrant:

```bash
docker compose up -d qdrant
VECTOR_STORE_PROVIDER=qdrant EMBEDDING_PROVIDER=openai OPENAI_API_KEY=sk-... \
    python -m app.knowledge.ingest --reset
```

`--reset` is required the first time you switch from `fake` to `openai` (or between different
OpenAI models with different output widths): the existing collection was created for whichever
embedding dimension was in use at the time (8 for `FakeEmbeddingProvider`; 1536 for
`text-embedding-3-small`), and Qdrant rejects vectors of the wrong width for an existing
collection outright rather than silently reinterpreting them - `--reset` recreates the collection
at the new, correct size before ingesting. `_embedding_dimensions_for()` resolves that size from
`get_embedding_provider(settings).dimensions` (the Phase 2.6 fix), so switching
`EMBEDDING_PROVIDER` alone is enough - no separate dimension setting to update by hand.

`get_embedding_provider(settings)` fails fast and clearly (`RuntimeError`, before any embedding
or Qdrant call) if `EMBEDDING_PROVIDER=openai` is set without `OPENAI_API_KEY` -
`app/knowledge/ingest.py` catches this and reports `Configuration error: ...` / `Status: FAILED`
rather than letting an unhandled exception surface partway through ingestion.

`FakeEmbeddingProvider` remains the default and is untouched - all existing tests, and every
local/offline workflow that doesn't need semantic retrieval quality, keep working exactly as
before. Nothing about the `EmbeddingProvider`/`VectorStore` Protocols, `IngestionService`, or the
ingestion command's structure changed to support this - the existing abstractions already
supported real embeddings the moment Phase 2.6 fixed dimension resolution; this phase is choosing
to actually use them for the persistent Qdrant store, not adding new plumbing.

## RAG / grounded answering

```
Retriever  →  VectorSearchResult[]
                    ↓
AnswerGenerator (Protocol-free, like ChatService)
  ├── build_context()        →  deterministic text block: document title + section heading + chunk text
  ├── build_history_block()   →  deterministic text block: prior conversation turns (may be empty)
  ├── build_prompt()           →  grounding instructions + history? + context + question
  └── LLMProvider.generate(LLMRequest(messages=[...]))  →  LLMResponse.text
                    ↓
GroundedAnswer { answer, sources[] }
```

- **`AnswerGenerator`** (`app/rag/answer_generator.py`) depends only on the `Retriever` and
  `LLMProvider` Protocols - never OpenAI, pgvector, ConversationStore, ToolRegistry, or FastAPI
  directly. It has no Protocol of its own: like `ChatService`, it's application orchestration
  logic with exactly one real implementation, not an external boundary with swappable backends -
  testability already comes from `Retriever`/`LLMProvider` each being fakeable. Unchanged in
  behavior by Step 15's `LLMProvider.generate()` evolution: it never sets `LLMRequest.tools`, so
  its one LLM call can never receive a tool-call response back - see Agent / tool calling for
  why `AgentService`, not `AnswerGenerator`, is what actually offers tools, and why
  `AnswerGenerator` was reused rather than modified or removed.
- **Context construction** (`build_context`) is a small, pure, directly-tested function. It
  includes only `document_title`, `section_heading`, and `text` per retrieved chunk - not
  `score`/`id`/`position`, which mean nothing to the model and would just be prompt noise.
  Same input always produces the same string.
- **Conversation history** (Step 13): `answer(question, history: list[ConversationMessage] =
  None)` gained an optional parameter - not a new constructor dependency. History is *data*
  passed per call, not a boundary `AnswerGenerator` depends on; its own dependency graph
  (`Retriever`, `LLMProvider`) is exactly what it was in Step 11. `build_history_block` formats
  prior turns as `"User: ...\nAssistant: ..."`; `build_prompt` omits the "Conversation so far"
  section entirely when there's no history, so Step 11's original prompt shape (and its
  existing tests) are unchanged for a first turn.
- **Grounding**: a fixed, concise system instruction (not configuration - see Configuration
  below for why) tells the model to answer only from the supplied knowledge context, never
  invent policies/prices/requirements/availability, and say so explicitly when the context is
  insufficient - plus one more sentence added in Step 13: conversation history is for
  understanding what's already been discussed, never a source of policies/prices/requirements/
  availability itself, only the knowledge context is. The composed string
  (grounding instructions + history + context + question, built by `build_prompt()`) still
  becomes a single `LLMMessage` - Step 15 changed how that message reaches the model
  (`LLMProvider.generate()` instead of the retired `generate_reply()`), not what's in it.
- **Empty retrieval**: if `Retriever.retrieve()` returns no results, `AnswerGenerator` returns
  a fixed `NOT_AVAILABLE_ANSWER` and **never calls the LLM** - tested explicitly. The
  application layer decides "we have nothing relevant," not the model.
- **Similarity threshold**: unchanged from Step 10 - `AnswerGenerator` does not add a second
  threshold. `Retriever`'s `min_score` (still off by default) remains the only place that
  decides which results are relevant enough to use.
- **`ChatService` integration**: as of Step 12 (and unchanged in shape by Step 13),
  `ChatService` delegates to `AnswerGenerator` (see Chat API integration below) rather than
  folding retrieval into `ChatService` directly - `ChatService` would otherwise duplicate
  orchestration `AnswerGenerator` already owns, and the provider-agnostic `LLMProvider`
  boundary stays cleanest when exactly one thing composes it for grounded answers
  (`AnswerGenerator`) and `ChatService` just calls that, now also coordinating
  `ConversationStore` alongside it.

## Chat API integration

```
POST /api/v1/chat
  ↓
ChatRequest { message, conversation_id? } (validated)
  ↓
ChatService.get_reply()  →  ConversationStore.get/create + get_recent_messages
  ↓
AgentService.answer(message, history)  →  GroundedAnswer { answer, sources[] }
  (internally: retrieval, up to one bounded tool-call round, final LLM call - see
  Agent / tool calling below)
  ↓
ChatService  →  ConversationStore.append_message() x2 (user, then assistant)
  ↓
ChatResponse { reply, sources[], conversation_id }
```

- **`ChatService`** (`app/chats/service.py`) now holds an `AgentService` *and* a
  `ConversationStore` - it knows WHAT it needs (an answer to a message, in the context of a
  conversation), never that retrieval, embeddings, vector search, tool execution, an LLM SDK, or
  conversation persistence are involved. `get_chat_service()` is the one FastAPI-`Depends()`-wired
  seam in this whole chain: it resolves `Settings` via `Depends(get_settings)`, then calls the
  plain `get_agent_service(settings)`/`get_conversation_store(settings)` composition functions -
  the same bridge pattern every other layer's composition function already documented as its
  own eventual FastAPI entry point. `AgentService`'s result type (`GroundedAnswer`) is identical
  to what `AnswerGenerator` returned, so this was the only real change `ChatService` needed for
  Step 15.
- **Request/response contract** (see Conversation context section for the full decision):
  `ChatRequest` gained an optional `conversation_id`; `ChatResponse` gained
  `conversation_id: str` alongside the existing `reply: str` and `sources: list[AnswerSource]`
  (`AnswerSource` still reused directly from `app/rag/models.py`, not redefined). A client that
  ignores `conversation_id` entirely still works exactly as before - each request just starts
  and immediately ends its own new conversation.
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

## Conversation context

```
Conversation { id, messages[] }
ConversationMessage { role, content, created_at }
        ↓
ConversationStore (Protocol)
        ↑
InMemoryConversationStore
```

**Knowledge base vs. conversation memory - kept strictly separate.** The knowledge base
(`app/knowledge`) is car-rental policies/services/documents/embeddings/vector search - facts
about the business. Conversation memory (`app/conversation`) is which messages were exchanged,
in what order, in which conversation - a record of a dialogue, not a fact about car rentals.
Conversation messages are never embedded and never written to `VectorStore`; the two domains
share no model, no table, no store.

**Scope decision: `ConversationStore` Protocol + `InMemoryConversationStore` only** - no
PostgreSQL-backed implementation yet. Considered:
1. In-memory store only (chosen)
2. PostgreSQL-backed persistence now
3. Some other approach

Chosen (1), for the same reason every other boundary in this codebase started this way
(`LLMProvider`, `EmbeddingProvider`, `VectorStore`): build the Protocol and prove the actual
use case first, add a real backend once something concretely needs it. Nothing does yet - there
is no authentication, no multi-process deployment, and no requirement for a conversation to
outlive the process. An in-process store is an honest match for what this stage needs, not a
shortcut around a "real" implementation. The Protocol is already shaped for a future
`PgVectorStore`-style implementation reusing Step 9's exact SQLAlchemy Core pattern (a table
keyed by conversation id and message order) - the trade-off of deferring it is that a process
restart currently loses all conversations, acceptable for a local-development-stage prototype
but not for production, which is explicitly out of scope for this step.

**Models** — deliberately minimal: `ConversationMessage` is `role` (`"user"` | `"assistant"`),
`content`, `created_at` (needed so a future SQL-backed store has something real to `ORDER BY` -
list position alone doesn't survive a database round-trip the way it does in memory).
`Conversation` is just `id` (a random UUID, not sequential/guessable - server-generated, a
client never invents one) plus its `messages`. No user/auth ids, no rental/booking ids, no
analytics, no token accounting, no tool-call state, no message embeddings - all later steps.

**`ConversationStore`** exposes exactly four operations, not generic CRUD: `create()`, `get()`
(returns `Optional[Conversation]` - a normal, expected outcome for an unknown id, not an
error), `append_message()`, and `get_recent_messages()` (oldest-first, ready to read top to
bottom into a prompt). The latter two raise `ConversationNotFoundError` for an unknown id,
since by the time either is called `ChatService` has already established the conversation
exists - a genuine bug if this ever surfaces at the HTTP layer, so it isn't mapped to any
response status (see Chat API integration's error handling). Like `VectorStore`'s in-memory
implementation, `get_conversation_store()` returns a process-wide singleton for the "memory"
provider (otherwise one request's appended messages would be invisible to the next).

**Memory strategy - bounded recent-message window, nothing more.** `ChatService` loads the
last `conversation_history_window` messages (default `6`, ~3 user/assistant turns) before
calling `AnswerGenerator`, and appends the new user+assistant messages afterward. No
summarization, no long-term/semantic memory - explicitly out of scope for this step, and
premature before there's a real usage pattern to design either against.

**RAG interaction - retrieval stays focused on the current question.** Conversation history is
never concatenated into the vector-search query; `Retriever.retrieve()` still receives only the
bare current message, unchanged from Step 10 (verified by
`test_retrieve_is_called_with_only_the_current_question_not_history`). History only enters the
*LLM prompt*, as its own labeled section, separate from the knowledge context - a deliberate,
minimal choice, not an oversight: naively appending prior turns to the search query would drift
the query away from the user's actual current information need (e.g. a short follow-up like
"and what about deposits?" would search on the whole conversation's text, not just "deposits").
Rewriting a follow-up into a standalone, retrieval-friendly query is real, valuable, and
explicitly deferred - a future step's job once there's a concrete case to design it against.

**Chat API contract.** `ChatRequest.conversation_id` is optional; omit it to start a new
conversation, supply a previously-returned one to continue it. `ChatResponse.conversation_id`
is always present, so the client always knows which id to use next - including after an
unknown/stale id silently starts a fresh conversation instead of erroring (a `ChatService`
policy decision, not a `ConversationStore` one: the store's `get()` just reports "not found";
what to *do* about that is the application layer's call). `Conversation.id` (a random UUID) is
used directly as the public identifier - there is no separate internal-vs-public id layer,
because nothing today (no auth, no multi-tenancy) needs one; a real internal database key,
when one exists, would need hiding for different reasons than a UUID already satisfies.

## Tools

```
Tool (Protocol): metadata, execute(raw_input: dict)
  ↑
CheckVehicleAvailabilityTool
  ↓
BusinessServiceClient (Protocol)
  ↑
FakeBusinessServiceClient          (today)
HTTP client → NestJS Business API  (future, not built)
```

**RAG/knowledge base vs. tools - the core distinction this step establishes:**

| | Knowledge base (`app/knowledge`, RAG) | Tools (`app/tools`) |
|---|---|---|
| Answers | Relatively static knowledge: policies, services, vehicle-use rules, requirements, general pricing *stated in documents* | Dynamic/live business operations: vehicle availability, booking lookup/creation/modification, customer-specific data |
| Source of truth | The six canonical PDFs, via retrieval | The business system (future: NestJS Business API) |
| Mechanism | Embedding + vector similarity search | Structured, validated function-style calls |

This Python service must never become the owner of rental business data. `app/tools` is
explicitly an **integration boundary**, not a business-domain owner - it holds no rental
inventory, no booking state, no customer records, no authentication, no authorization, no
pricing rules. All of that belongs to the business backend, today represented only by a fake.

**Scope (Step 14): established the boundary, not the agent.** Built: `Tool` Protocol,
`ToolMetadata`, `ToolRegistry`, one concrete tool (`check_vehicle_availability`),
`BusinessServiceClient` Protocol + `FakeBusinessServiceClient`, and tests proving all of it. Not
built then: LLM tool/function calling or the real NestJS HTTP client. Step 15 (see Agent / tool
calling below) built the former - a bounded, provider-agnostic tool-calling agent - while
deliberately still not building the latter or an unrestricted autonomous loop.

**`Tool` Protocol** (`app/tools/tool.py`): `metadata: ToolMetadata` (name, description,
`input_schema: dict`) and `async execute(raw_input: dict) -> BaseModel`. `execute` takes a
plain, unvalidated dict - the shape an LLM's function-call arguments would actually arrive in -
and returns each tool's own result model. There is no generic `ToolResult` wrapper: a
`success`/`data`/`error` envelope would erase what's actually different between tools' results,
and this codebase already prefers "return the real type, raise on failure" everywhere else
(`Retriever`, `AnswerGenerator`, ...) over a result-wrapper pattern.

**Tool metadata is provider-agnostic on purpose.** `input_schema` is plain JSON Schema
(`InputModel.model_json_schema()`), not OpenAI's `{"type": "function", "function": {...}}`
envelope or any other provider's specific wrapper - JSON Schema is what every major
function-calling API already expects parameters shaped like, so no adaptation is needed to hand
this to one, but nothing here commits to a specific provider either.

**`ToolRegistry`** (`app/tools/registry.py`): `register()`, `resolve()` (returns
`Optional[Tool]` - an unknown name is a normal, expected lookup outcome, matching
`ConversationStore.get()`'s precedent, not an error the registry itself raises), and
`list_tools()` (returns `ToolMetadata`, not tool instances - a future agent needs to see what's
available, not reach into a tool's internals). Deliberately not a plugin system: tools are
registered explicitly in code (`get_tool_registry()`), never discovered via dynamic imports or
runtime code loading, so the available-tools list is always the same deterministic set for a
given process.

**`check_vehicle_availability`** (`app/tools/check_vehicle_availability.py`): input
`pickup_at`, `return_at` (both required `datetime`s; `return_at` must be strictly after
`pickup_at` - validated, tested), optional `category` (free-text, not an enum - a vehicle
category taxonomy is business inventory knowledge, not something this service should hardcode).
No duration cap or other business-rule validation was added beyond ordering: "how long can a
rental be" is a business policy question for the backend to own, not a sanity check Python
should guess at. Result: `available_vehicles: list[VehicleAvailability]`
(`vehicle_id`, `category`, `model`, `available_from`, `available_until`) - only what a caller
needs to know availability, no database-specific fields.

**`BusinessServiceClient`** (`app/tools/business_client.py`): one operation today,
`check_vehicle_availability(start, end, category=None)`, because one tool exists; a future
booking-lookup/creation tool would add its own method here when it's actually needed, not
speculatively now. `FakeBusinessServiceClient` is a deterministic test double, not a second
business database - it holds only whatever fixed vehicle list a test passes in (default: none),
filtering by date-range containment and category; nothing is ever *written* to it at runtime,
so there is no pricing, booking, or customer logic to duplicate. The real HTTP client calling
the NestJS Business API is **not built this step** - nothing in this repository consumes it yet
(no agent, no endpoint), and building it now would mean guessing at NestJS's request/response
shapes and auth before there's a concrete contract to build against; `get_business_service_client()`
already selects by `Settings.business_service_provider`, so adding it later touches no tool.

**Errors** (`app/tools/exceptions.py`) - a small, three-member hierarchy under `ToolError`,
matching exactly what a future agent needs to distinguish: `ToolInputError` (raw input failed
the tool's own schema validation), `BusinessServiceUnavailableError` (the business system
couldn't be reached - raised by `BusinessServiceClient` implementations and left unwrapped by
the tool, the same "don't re-wrap an already-translated error" pattern `Retriever` uses for
`EmbeddingProviderError`/`VectorStoreError`), and `ToolExecutionError` (reserved for a tool
execution failure that is neither of the other two - not actively raised by
`check_vehicle_availability` today, since its only failure modes so far are exactly the other
two; kept in the hierarchy for a future tool that needs it).

**Composition**: `get_business_service_client(settings)` and `get_tool_registry(settings)` are
plain functions, not FastAPI `Depends()`-wired - `AgentService`'s own composition function
(`get_agent_service()`) calls `get_tool_registry()` directly, the same plain-function-calling-
plain-function pattern used throughout; no endpoint calls `get_tool_registry()` itself, only
`get_chat_service()` does, transitively. Unlike `get_vector_store()`/`get_conversation_store()`,
`get_tool_registry()` does **not** return a process-wide singleton: nothing ever writes through
`FakeBusinessServiceClient` after construction, so unlike Step 12's `InMemoryVectorStore` bug,
there is no state a later call could fail to see - a fresh registry per call is simply harmless
here.

**LLM integration boundary - built this step.** Step 14 deliberately deferred this; Step 15 is
where it happens. See the Agent / tool calling section below for the full design: how
`LLMProvider` evolved, how `AgentService` uses `ToolRegistry` to offer and execute tools, and
the bounded one-round policy that keeps this from becoming an unrestricted agent loop.

## Agent / tool calling

```
User
  ↓
AgentService
  ├── Retriever → knowledge context (unchanged from Step 11 - see Tool + RAG interaction below)
  └── LLMProvider.generate(LLMRequest{messages, tools: ToolRegistry.list_tools()})
        ├── LLMResponse{text}         → done, return it
        └── LLMResponse{tool_calls}   → execute via ToolRegistry (bounded: exactly one round)
                                          → LLMProvider.generate(..., tools=[])  → final text
  ↓
GroundedAnswer { answer, sources[] }
```

**Provider-agnostic LLM contract.** `LLMProvider.generate_reply(message: str) -> str` (Steps
3-14) is **retired**, replaced by `generate(request: LLMRequest) -> LLMResponse`
(`app/llm/provider.py`). One unambiguous contract was chosen over keeping both methods side by
side: `AnswerGenerator`'s change was a single, behavior-preserving call-site update (build an
`LLMRequest` with one user message, read `.text` back - see RAG section), so there was no real
case for carrying the old method forward as permanent legacy sugar. New application-level
models (`app/llm/models.py`), none of them OpenAI-specific:

```python
class ToolCall(BaseModel):
    id: str
    tool_name: str
    arguments: dict

class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_call_id: Optional[str] = None       # set on role == "tool": which ToolCall this answers
    tool_calls: list[ToolCall] = []           # set on role == "assistant" when tools were requested

class LLMRequest(BaseModel):
    messages: list[LLMMessage]
    tools: list[ToolMetadata] = []            # reused directly from app/tools - see below

class LLMResponse(BaseModel):
    text: Optional[str] = None
    tool_calls: list[ToolCall] = []
```

`LLMRequest.tools` reuses `ToolMetadata` (Step 14) **directly** - not duplicated, not adapted
into a parallel shape at this layer. `app/llm` importing from `app/tools` is a deliberate,
narrow exception to this codebase's usual "layers don't depend sideways" instinct: `ToolMetadata`
was explicitly designed in Step 14 to be "a provider-agnostic tool description a future
LLM/function-calling layer can be given" - this is exactly that layer, and the alternative
(a second, structurally-identical model plus a mapping function) would be pure duplication for
no benefit. `LLMResponse` guarantees exactly one of `text`/`tool_calls` is meaningful - both
`OpenAIProvider` and `FakeLLMProvider` uphold this, so `AgentService` never has to guess which
one to trust.

**`ToolCall`** is *not* the same thing as calling `Tool.execute()` directly - it's what the
model said it wants, before anything has been validated or run. `ToolCall.arguments` maps 1:1
onto `Tool.execute(raw_input: dict)`; there was no need for a second "pending call" wrapper
type beyond that.

**`AgentService`** (`app/agent/service.py`) is the new orchestration layer. It reuses
`AnswerGenerator`'s `build_context`/`build_prompt`/`to_source`/`NOT_AVAILABLE_ANSWER` (`to_source`
was made public - renamed from `_to_source` - specifically so this reuse wouldn't require
duplicating it) rather than reimplementing prompt construction, but needs its **own** top-level
method: tool-calling requires inspecting the LLM's response *before* deciding whether to return
it or execute a tool and continue - something `AnswerGenerator.answer()`'s single,
non-branching call cannot express, no matter how it's refactored, without giving it the same
tool-awareness `AnswerGenerator` is deliberately kept free of. `AnswerGenerator` itself is
**unchanged** and remains a fully working, directly usable, pure-RAG (no tool awareness)
building block - not modified, not deleted, not bypassed in the sense of being made unreachable
or untested; `ChatService` simply now depends on `AgentService` (a strict superset of its
capability) instead, the same kind of evolution as Step 12 switching `ChatService` from
`LLMProvider` to `AnswerGenerator`.

**Bounded execution policy: exactly one tool-call round.** Not a counter compared against a
configurable limit - a **structural** guarantee. If the model's first response requests tools,
`AgentService` executes them, appends the results, and makes exactly one follow-up call with
`tools=[]`: no tools offered at all, so the model cannot possibly request another one on that
call - the bound is enforced by what the model is offered, not by trusting it to stop. If a
follow-up response somehow still contains `tool_calls` anyway (only possible from a
provider that doesn't honor an empty tools list), `AgentService` raises `LLMProviderError`
rather than looping - failing loudly instead of silently allowing a second round. `1` is a
code constant (`MAX_TOOL_ROUNDS` in `app/agent/service.py`), not a setting - see Configuration
for why.

**RAG + tool interaction.** Retrieval always runs first and is **never** skipped or replaced by
a tool call - `AgentService.answer()` calls `Retriever.retrieve(question)` exactly like
`AnswerGenerator` does, with the bare current question only (verified by
`test_retrieval_is_called_with_only_the_current_question`). Both the retrieved knowledge context
*and* the available tools are offered to the model in the same first call, so:
- **Static knowledge** ("What is your cancellation policy?") → the model answers from the
  knowledge context in the prompt; no tool call needed.
- **Dynamic business data** ("Is a Camry available tomorrow?") → the model requests
  `check_vehicle_availability`; the knowledge context is present but likely unused for that turn.
- **Mixed requests** ("What is your cancellation policy, and is a Camry available tomorrow?") →
  both are already available to the model in the same call (knowledge context from retrieval,
  tool result once executed), so a reasonably capable model can address both without any extra
  planning code on this service's side. No keyword matching, heuristics, or hand-coded routing
  decide RAG-vs-tool in Python - that decision is the model's, made from the tool's own
  description (see Tools section) and the retrieved context, exactly as real function-calling is
  meant to work. Whether a given model actually does this well for a genuinely mixed request is
  a model-quality question, not an architecture gap - no complex planning was built to compensate
  for it, per this step's explicit scope.

**`FakeLLMProvider`** (`app/llm/fake_provider.py`) gained a `responses` parameter: an explicit,
ordered list of `LLMResponse` objects to return one per call (e.g. a tool-call response
followed by a final-text response), never inferred or "smart" - popping a plain list. With no
`responses` configured, it behaves exactly as before Step 15 (single-call echo of the most
recent user message, `[fake-llm-reply] ...`), so every pre-existing test and default local-dev
behavior is unaffected.

**`OpenAIProvider`** (`app/llm/openai_provider.py`) is the only place the openai SDK's
tool-calling types are ever touched. Verified against the actual installed SDK's type
definitions (not assumed) before writing the translation:
- `LLMRequest.tools` → OpenAI's `[{"type": "function", "function": {"name", "description",
  "parameters"}}]`; omitted entirely (not sent as `tools=[]`) when no tools are offered,
  preserving Step 3/4's original request shape exactly for that case.
- A response's `message.tool_calls` → `list[ToolCall]`, parsing each `Function.arguments` JSON
  string into a dict; malformed JSON is translated into `LLMProviderError`, never left to
  surface as a raw `json.JSONDecodeError`.
- A prior assistant tool-call turn and its tool-result turn are each translated back into
  OpenAI's specific message shapes (`role: "assistant"` with a `tool_calls` array,
  `role: "tool"` with `tool_call_id`) for the follow-up request - required for OpenAI's API to
  accept the conversation at all, since a `"tool"`-role message must reference a preceding
  assistant tool call by id.
- `message.content is None` with no tool calls still raises `LLMProviderError`, unchanged from
  Step 4.

**Error handling.** `_execute_tool_call` in `AgentService` catches every case explicitly:
unknown tool name (`ToolRegistry.resolve()` returns `None`), `ToolInputError`,
`BusinessServiceUnavailableError`, and `ToolExecutionError` - each translated into a short,
fixed, generic string (e.g. `"Error: the business service is currently unavailable."`) fed back
to the model as the tool's result, **never** the raw exception message (which could contain a
connection string or other internal detail). This lets the model produce a graceful final
answer ("I couldn't check availability right now") instead of the whole request failing outright
- more robust than treating every tool failure as fatal, and still fully safe: only a
`ToolRegistry`-**resolved** tool's own `execute()` is ever called, so arbitrary model output can
never execute arbitrary Python (see Tools section). `LLMProviderError` from the LLM call itself
(as opposed to a tool) is **not** caught here - it propagates unwrapped, exactly like every
other already-translated exception in this codebase, and is already mapped to a clean `503` by
the existing `app/api/exception_handlers.py` handler from Step 12; no changes were needed there.
Tool calls are **not** retried automatically anywhere in this flow.

**Chat API impact - none, by design.** `ChatRequest`/`ChatResponse` are byte-for-byte unchanged
from Step 13: `message`/`conversation_id` in, `reply`/`sources`/`conversation_id` out. A client
cannot tell from the response whether a tool was used - no tool-call trace, no intermediate
message list, no debug field, matching the requirement that tool use stays an internal
implementation detail of producing "a final grounded answer." `ConversationMessage` (Step 13)
is likewise **unchanged**: only the final user question and final assistant answer are ever
persisted; the intermediate tool-call/tool-result exchange lives entirely inside
`AgentService.answer()`'s local `messages: list[LLMMessage]` for that one request and is
discarded once it returns - never written to `ConversationStore`, never given a new
`ConversationMessage` role, exactly as scoped.

## LangChain integration

**Phase 2.7.1 status: dependency and integration boundary only - no behavior change.** Nothing
described elsewhere in this README changed: `LLMProvider`, `EmbeddingProvider`, `VectorStore`,
`VectorRetriever`, `IngestionService`, `AnswerGenerator`, and `AgentService` are all exactly as
Phases 1-2.6 left them, still what `/api/v1/chat` actually runs today.

**Why LangChain is being introduced:** later phases (2.7.2+) plan to build more elaborate
retrieval/generation chains and, eventually, a LangGraph-based conversation flow (Phase 4). Both
are large enough, and standard enough problems, that a maintained orchestration library is worth
depending on rather than hand-rolling - the same reasoning that already justified depending on
FastAPI, SQLAlchemy, or the OpenAI SDK instead of writing less than those from scratch.

**What LangChain will eventually handle** (not yet - future phases): composing multi-step
retrieval/generation chains declaratively, and - via LangGraph - the multi-turn booking
conversation flow (Understand → Search → Show Cars → Select → Collect Info → Confirm → Create
Booking) sketched in the wider project roadmap.

**What stays custom, indefinitely:** the provider boundaries themselves. `LLMProvider` and
`EmbeddingProvider` remain this project's own Protocols - LangChain components, when built, will
be composed *from* `OpenAIProvider`/`OpenAIEmbeddingProvider` or wrap the existing OpenAI SDK
calls, not replace the Protocol boundary that keeps the rest of the app decoupled from any one
SDK. `QdrantVectorStore` and the explicit ingestion command (`app/knowledge/ingest.py`) are
unaffected - Qdrant remains the vector database, populated exactly as Phase 2.6 built it,
regardless of what eventually queries it. `AgentService`'s bounded tool-calling and
`ConversationStore` are also unaffected for now. Business logic never moves into LangChain:
Strapi/PostgreSQL remains the sole source of truth for Cars/Bookings/Payments, and LangChain
components (when built) call into this project's own abstractions to reach it, the same as
everything else does.

**Integration boundary chosen:** `app/langchain_integration/` - a new, currently-empty-of-logic
package (just a docstring establishing its purpose) where LangChain-specific code will live once
built. Named `langchain_integration`, not `langchain`, so a local package never shadows the real
`langchain` import anywhere under `app/`. This mirrors every existing SDK boundary in this
project: the `openai` SDK is isolated to `app/llm/openai_provider.py` and
`app/knowledge/openai_embedding_provider.py`; `qdrant-client` to
`app/knowledge/qdrant_vector_store.py`; `pgvector`/SQLAlchemy to `app/knowledge/pgvector_store.py`.
`langchain`/`langchain-openai`/`langchain-qdrant` are meant to stay isolated to
`app/langchain_integration/` alone in exactly the same way - nothing elsewhere in `app/` imports
them.

**Dependencies added:** `langchain`, `langchain-openai`, `langchain-qdrant` (plus their own
transitive dependencies - notably `langchain-core`, `langsmith`, `tiktoken`). No other LangChain
package was added; `langgraph` is deferred to Phase 4, when it's actually used.
`tests/langchain_integration/test_imports.py` verifies all three import cleanly in this project's
environment (Python 3.9, alongside the existing `openai`/`qdrant-client`/pydantic v2 stack) - it
constructs a `langchain_core.documents.Document` and checks `ChatOpenAI`/`OpenAIEmbeddings`
subclass the expected LangChain base classes, but builds no chain and calls no API.

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
- [x] Conversation context foundation (`ConversationStore`, bounded recent-message window,
      multi-turn `/api/v1/chat` - in-memory only, single-process)
- [x] Tool capability boundary (`Tool`, `ToolMetadata`, `ToolRegistry`,
      `check_vehicle_availability`, `BusinessServiceClient` - fake implementation only)
- [x] Provider-agnostic tool calling / agent orchestration foundation (`AgentService`,
      `LLMProvider.generate()`, bounded to exactly one tool-call round)
- [x] Canonical six-document knowledge base (`data/knowledge/`), replacing the original two PDFs
- [x] `QdrantVectorStore` - Qdrant as this project's AI knowledge vector store, alongside the
      existing `InMemoryVectorStore`/`PgVectorStore` (`PgVectorStore` kept, not the intended
      store going forward)
- [x] Persistent, explicit, idempotent Qdrant ingestion command (`python -m app.knowledge.ingest`,
      `--reset` for stale-document rebuilds) - separate from the in-memory-only startup bootstrap
- [x] Embedding-dimension resolution fixed to derive from the actual configured EmbeddingProvider,
      not an OpenAI-only assumption (`pgvector` and `qdrant` can no longer be sized incorrectly)
- [ ] Real OpenAI embeddings wired end-to-end with Qdrant - **code/config path confirmed ready
      (Phase 2.7.2)**, actual ingestion still blocked on an `OPENAI_API_KEY` not being available
      in this environment; not yet run for real, see Known limitations
- [x] LangChain dependencies added (`langchain`, `langchain-openai`, `langchain-qdrant`) and
      `app/langchain_integration/` established as their sole isolation boundary - no chain built,
      no existing behavior changed (Phase 2.7.1)
- [ ] First real LangChain component (a retrieval/generation chain) built in
      `app/langchain_integration/`
- [ ] PostgreSQL-backed conversation persistence
- [ ] Long-term/semantic conversation memory, summarization, query rewriting for follow-ups
- [ ] Real NestJS Business API integration (HTTP `BusinessServiceClient`)
- [ ] More than one tool-call round, parallel tool execution, additional tools beyond
      `check_vehicle_availability`
- [ ] Autonomous planning / multi-agent systems
- [ ] Booking creation, modification, or cancellation
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
- **Framework-independent knowledge, RAG, conversation, tools, and agent layers** — none of
  `app/knowledge`, `app/rag`, `app/conversation`, `app/tools`, or `app/agent` has a FastAPI
  dependency, so all five are usable (and testable) outside a request/response cycle.
- **Evolve a contract rather than duplicate it** — when `LLMProvider` needed to support tool
  calls (Step 15), the old `generate_reply()` was retired in favor of one richer `generate()`
  method, not kept alongside it as permanent legacy sugar. The one call site that used it
  (`AnswerGenerator`) got a small, behavior-preserving update instead.
- **Separate domains stay separate** — the knowledge base (car-rental facts, embeddings, vector
  search), conversation memory (dialogue identity, ordering, history), and tools (dynamic
  business operations, owned by the business backend) share no model, table, or store.
- **Integration boundaries aren't business-domain owners** — `app/tools` describes and invokes
  business operations; it holds no rental inventory, booking state, customer records,
  authentication, or pricing rules. That data stays owned by the (future) NestJS Business API.
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
