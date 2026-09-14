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
  embeddable units: extraction (`DocumentExtractor`), chunking (`DocumentChunker`), and
  embedding (`EmbeddingProvider`), each behind its own Protocol. Framework-independent: it
  doesn't import FastAPI and isn't wired into any endpoint yet.

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
- Unit/integration tests

**Not implemented yet:**
- Vector database
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
├── knowledge/         # Document contracts, extraction, chunking, embedding providers
└── llm/               # LLMProvider Protocol + fake/OpenAI implementations

tests/                # Mirrors the app/ layout, one test package per feature
data/
└── knowledge/        # Source PDFs for the future knowledge base
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
```

**Planned:**
```
embedding vector[]
  ↓
Vector Store
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
statistics against the real PDFs), and the embedding provider abstraction (fake, and OpenAI
with a mocked SDK client). All tests run without any external network access or API key.

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

## Roadmap

- [x] FastAPI foundation
- [x] Chat API
- [x] LLM provider abstraction
- [x] OpenAI provider
- [x] PDF extraction foundation
- [x] Document chunking
- [x] Embedding provider
- [ ] Vector store
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
- **Avoiding unnecessary infrastructure** — no database, cache, queue, or vector store has
  been introduced before the feature that actually needs it.
