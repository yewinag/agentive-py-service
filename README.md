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
  ├── Chat                      (app/chats)
  ├── Conversation Memory         (app/conversation)
  ├── RAG / Answering               (app/rag)
  ├── Tools                           (app/tools)  →  (future) NestJS Business API
  ├── LLM Provider                       (app/llm)
  └── Knowledge Pipeline                   (app/knowledge)
```

Six boundaries, each with a single responsibility:

- **Chat/application layer** (`app/chats`) — orchestrates one HTTP request: validates input,
  resolves/continues a conversation, delegates to `AnswerGenerator` via `ChatService`, shapes
  the response. `POST /api/v1/chat` returns a knowledge-grounded, multi-turn-aware answer. Not
  yet connected to `app/tools` - see Tools section for why.
- **Conversation memory layer** (`app/conversation`) — a domain deliberately separate from the
  knowledge base (see Conversation context section below): identity, ordering, and bounded
  recent-history for a conversation, behind `ConversationStore`. Holds no car-rental knowledge,
  no embeddings, no vectors.
- **RAG / grounded-answering layer** (`app/rag`) — combines retrieval, bounded conversation
  context, and LLM generation into one grounded answer, behind `AnswerGenerator`. Depends on
  the `Retriever` and `LLMProvider` Protocols only. `ChatService` is its one real consumer.
- **Tools layer** (`app/tools`) — a domain deliberately separate from both the knowledge base
  and the business backend itself (see Tools section below): describes and executes dynamic
  business operations (today: vehicle availability) behind a `Tool` Protocol and a
  `ToolRegistry`, talking to the business system through a `BusinessServiceClient` Protocol.
  Owns no rental inventory, booking state, or customer data - it's an integration boundary, not
  a business-domain owner. Not yet connected to any LLM/agent decision loop or any endpoint.
- **LLM provider layer** (`app/llm`) — answers "how do we generate a reply?" behind an
  `LLMProvider` Protocol, so the concrete provider (a fake, OpenAI, or anything else later)
  is swappable without touching the chat, conversation, RAG, or tools layers.
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
- Unit/integration tests

**Not implemented yet:**
- Conversation persistence beyond a single process (no PostgreSQL-backed `ConversationStore`
  yet - see Conversation context section for why)
- Long-term/semantic conversation memory, summarization, or query rewriting for follow-ups
- Real NestJS Business API integration (no HTTP `BusinessServiceClient` yet - see Tools section)
- LLM tool/function calling - the LLM cannot invoke a tool yet; `ToolRegistry` exists, nothing
  decides when to use it
- Agent orchestration / autonomous planning
- Booking creation, modification, or cancellation of any kind
- Q&A evaluation/improvement loop
- Streaming, reranking, hybrid search
- Authentication, business/rental database integration
- Production deployment

## Project structure

```
app/
├── api/            # Router aggregation + shared exception handlers
├── chats/           # Chat feature: router, schemas, ChatService (-> AnswerGenerator + ConversationStore)
├── conversation/     # Conversation, ConversationMessage, ConversationStore Protocol + in-memory impl
├── core/             # Cross-cutting config (Settings)
├── health/           # Health-check endpoint
├── knowledge/         # Document contracts, extraction, chunking, embedding, storage,
│                       # retrieval, ingestion orchestration, and startup bootstrap
├── llm/               # LLMProvider Protocol + fake/OpenAI implementations
├── rag/                # AnswerGenerator: Retriever + LLMProvider + history -> grounded answer
└── tools/               # Tool/ToolRegistry, BusinessServiceClient, check_vehicle_availability

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

**Retrieval + RAG + conversation context, reachable via `POST /api/v1/chat`:**
```
user message + optional conversation_id
  ↓
ChatService  →  ConversationStore  (resolve conversation, load bounded recent history)
  ↓
AnswerGenerator  →  Retriever  →  EmbeddingProvider  →  VectorStore.search()
  ↓                                      ↓
LLMProvider.generate_reply()   ranked VectorSearchResult[]
  (prompt = grounding + history + knowledge context + question)
  ↓
ChatService  →  ConversationStore  (append user + assistant messages)
  ↓
ChatResponse { reply, sources[], conversation_id }
```

Every stage is implemented, tested, and wired end to end. Ingestion still has no HTTP
trigger - the default in-memory store is populated by an explicit startup bootstrap instead
(see Chat API integration below). Tools, agent orchestration, conversation summarization, and
durable (cross-process) conversation persistence remain future work.

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
- `POST /api/v1/chat` — returns a knowledge-grounded, multi-turn-aware answer, by default drawn
  from the two PDFs bootstrapped into the in-memory store at startup (see Chat API integration
  and Conversation context below)

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

VECTOR_STORE_PROVIDER=memory         # "memory" or "pgvector"
DATABASE_URL=postgresql+asyncpg://agentive:agentive@localhost:5432/agentive  # only for pgvector

RETRIEVAL_TOP_K=5                    # default VectorRetriever.retrieve() top_k
# RETRIEVAL_MIN_SCORE=0.75           # unset by default - see Retrieval section below

CONVERSATION_STORE_PROVIDER=memory   # only "memory" is implemented today
CONVERSATION_HISTORY_WINDOW=6        # last N messages (~3 turns) sent to the LLM as context

BUSINESS_SERVICE_PROVIDER=fake       # only "fake" is implemented today - see Tools section
```

`.env` is gitignored and must never be committed — only `.env.example`, with placeholder
values, is tracked.

Ingestion and wiring chat to RAG needed no new settings. Step 13 added
`conversation_store_provider`/`conversation_history_window`; Step 14 adds
`business_service_provider`, the same shape again - a provider selector kept even with only one
working value today, so a real NestJS-calling client slots in later without touching any tool.
The grounding system prompt remains a fixed code constant, not a setting - it's not something
that should vary by environment, and making it configurable would just be a place for
inconsistent/untested prompt variants to creep in.

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
Retriever → FakeLLMProvider → AnswerGenerator → InMemoryConversationStore → ChatService →` the
real FastAPI app - including a two-request test proving a follow-up's prompt genuinely contains
the prior turn's exact question, not just that something was stored - and a parametrized test
confirming embedding/vector-store/LLM failures all return a clean 503 with no provider-specific
detail in the body. `ConversationStore` has its own dedicated coverage: creating a conversation,
appending messages in order, a bounded recent-message window (oldest-first, correctly truncated
and correctly returning everything when there's less than the limit), a missing conversation id
(`get()` returns `None`; `append_message`/`get_recent_messages` raise
`ConversationNotFoundError`), and isolation between two unrelated conversations.
`ChatService` itself has unit-level orchestration tests (stub `AnswerGenerator`, real
`InMemoryConversationStore`) proving a new chat creates a conversation, a follow-up reuses it,
both turns get appended, prior turns are actually passed as `history` to `AnswerGenerator`, an
unknown `conversation_id` falls back to a new conversation rather than erroring, and the
history window is respected. The tool boundary has its own coverage too: `ToolRegistry`
(register, resolve, list, unknown-name lookup, duplicate-registration rejection), the metadata/
input-schema/valid-execution/invalid-input/error-propagation contract for
`CheckVehicleAvailabilityTool`, `FakeBusinessServiceClient`'s deterministic date-range/category
filtering, provider-selection composition tests, and an integration test chaining
`FakeBusinessServiceClient → CheckVehicleAvailabilityTool → ToolRegistry`. All of the above run
without any external network access, API key, or database.

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
  ├── build_context()        →  deterministic text block: document title + section heading + chunk text
  ├── build_history_block()   →  deterministic text block: prior conversation turns (may be empty)
  ├── build_prompt()           →  grounding instructions + history? + context + question
  └── LLMProvider.generate_reply(prompt)  →  answer
                    ↓
GroundedAnswer { answer, sources[] }
```

- **`AnswerGenerator`** (`app/rag/answer_generator.py`) depends only on the `Retriever` and
  `LLMProvider` Protocols - never OpenAI, pgvector, ConversationStore, or FastAPI directly. It
  has no Protocol of its own: like `ChatService`, it's application orchestration logic with
  exactly one real implementation, not an external boundary with swappable backends -
  testability already comes from `Retriever`/`LLMProvider` each being fakeable.
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
  availability itself, only the knowledge context is. `LLMProvider.generate_reply()` still
  takes one plain string, unchanged since Step 3/4 - grounding instructions, history, context,
  and question are all composed into that single string by `build_prompt()`.
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
AnswerGenerator.answer(message, history)  →  GroundedAnswer { answer, sources[] }
  ↓
ChatService  →  ConversationStore.append_message() x2 (user, then assistant)
  ↓
ChatResponse { reply, sources[], conversation_id }
```

- **`ChatService`** (`app/chats/service.py`) now holds an `AnswerGenerator` *and* a
  `ConversationStore` - it knows WHAT it needs (an answer to a message, in the context of a
  conversation), never that retrieval, embeddings, vector search, an LLM SDK, or conversation
  persistence are involved. `get_chat_service()` is the one FastAPI-`Depends()`-wired seam in
  this whole chain: it resolves `Settings` via `Depends(get_settings)`, then calls the plain
  `get_answer_generator(settings)`/`get_conversation_store(settings)` composition functions -
  the same bridge pattern every other layer's composition function already documented as its
  own eventual FastAPI entry point.
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
| Source of truth | The two PDFs, via retrieval | The business system (future: NestJS Business API) |
| Mechanism | Embedding + vector similarity search | Structured, validated function-style calls |

This Python service must never become the owner of rental business data. `app/tools` is
explicitly an **integration boundary**, not a business-domain owner - it holds no rental
inventory, no booking state, no customer records, no authentication, no authorization, no
pricing rules. All of that belongs to the business backend, today represented only by a fake.

**Scope decision - established the boundary, not the agent.** Built: `Tool` Protocol,
`ToolMetadata`, `ToolRegistry`, one concrete tool (`check_vehicle_availability`),
`BusinessServiceClient` Protocol + `FakeBusinessServiceClient`, and tests proving all of it.
Not built: an autonomous agent loop, LLM tool/function calling, or the real NestJS HTTP client -
each explained below, deliberately deferred rather than a general-purpose agent framework built
prematurely.

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
plain functions, not FastAPI `Depends()`-wired - no endpoint consumes a `ToolRegistry` yet.
Unlike `get_vector_store()`/`get_conversation_store()`, `get_tool_registry()` does **not**
return a process-wide singleton: nothing ever writes through `FakeBusinessServiceClient` after
construction, so unlike Step 12's `InMemoryVectorStore` bug, there is no state a later call
could fail to see - a fresh registry per call is simply harmless here.

**LLM integration boundary - deliberately not built.** The existing `LLMProvider.generate_reply(message: str) -> str`
Protocol was not changed. Supporting real function-calling would need a richer contract (a way
to pass tool definitions to the model and receive back which tool it wants called, with what
arguments) that no current caller needs yet. Changing `LLMProvider` now, with nothing to
exercise the new shape, would be exactly the kind of premature abstraction this codebase has
consistently avoided (see Engineering principles). The target flow this step sets up for -
`User → Agent → LLM decides whether a tool is needed → ToolRegistry → Tool →
BusinessServiceClient → NestJS Business API` - is a later step's job.

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
- [ ] PostgreSQL-backed conversation persistence
- [ ] Long-term/semantic conversation memory, summarization, query rewriting for follow-ups
- [ ] Real NestJS Business API integration (HTTP `BusinessServiceClient`)
- [ ] LLM tool/function calling (the LLM deciding when to invoke a tool)
- [ ] Agent orchestration / autonomous planning
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
- **Framework-independent knowledge, RAG, conversation, and tools layers** — none of
  `app/knowledge`, `app/rag`, `app/conversation`, or `app/tools` has a FastAPI dependency, so
  all four are usable (and testable) outside a request/response cycle.
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
