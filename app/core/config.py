from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, sourced from environment variables / .env.

    Equivalent to NestJS's ConfigService: a single typed object injected
    wherever configuration is needed, instead of reading os.environ ad hoc.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Agentive AI Service"
    environment: str = "development"
    debug: bool = False

    # Selects the LLMProvider implementation in app/llm/dependencies.py.
    # One of: "fake", "openai".
    llm_provider: str = "fake"

    # Required only when llm_provider == "openai". No default on purpose -
    # a missing key must fail loudly rather than silently falling back.
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # Selects the EmbeddingProvider implementation in
    # app/knowledge/embedding.py. One of: "fake", "openai". Independent
    # of llm_provider - chat replies and embeddings are separate concerns
    # that happen to often use the same OpenAI account.
    embedding_provider: str = "fake"

    # Dedicated setting, not reused from openai_model: chat and embedding
    # are different OpenAI model families with different valid values.
    openai_embedding_model: str = "text-embedding-3-small"

    # Selects the VectorStore implementation in app/knowledge/vector_store.py.
    # One of: "memory", "pgvector", "qdrant".
    vector_store_provider: str = "memory"

    # Required only when vector_store_provider == "pgvector". No default
    # on purpose, same reasoning as openai_api_key. Expected form:
    # postgresql+asyncpg://user:password@host:5432/dbname
    database_url: Optional[str] = None

    # Used only when vector_store_provider == "qdrant". Unlike
    # database_url, a plain host:port carries no credentials, so a safe
    # local-dev default is reasonable (matches docker-compose.yml).
    qdrant_url: str = "http://localhost:6333"

    # Collection Qdrant stores chunk embeddings under. Vector width is
    # not separately configured here - both pgvector and Qdrant reuse
    # the embedding provider's own dimensions (see
    # vector_store.py:_embedding_dimensions_for), so the two can never
    # silently drift apart.
    qdrant_collection: str = "knowledge_chunk_embeddings"

    # Default VectorRetriever.retrieve() top_k, overridable per call.
    retrieval_top_k: int = 5

    # No default on purpose (None = no filtering): there is no empirical
    # basis yet for a universal similarity cutoff for a given embedding
    # model/knowledge base. See README's Retrieval section.
    retrieval_min_score: Optional[float] = None

    # Selects the ConversationStore implementation in
    # app/conversation/store.py. Only "memory" is implemented today; kept
    # as a setting (matching every other swappable boundary) so a future
    # PostgreSQL-backed store slots in without touching ChatService.
    conversation_store_provider: str = "memory"

    # Bounded conversation context sent to the LLM alongside the current
    # question: the last N messages (~history_window/2 user+assistant
    # turns). 6 = 3 turns - enough for short-term follow-up context
    # without unbounded prompt growth. Not calibrated against real usage
    # data (no evaluation loop exists yet - see README), so treated as a
    # safe, conservative default rather than a precisely tuned number.
    conversation_history_window: int = 6

    # Selects the BusinessServiceClient implementation in
    # app/tools/business_client.py. One of: "fake", "strapi". Defaults to
    # "fake" so existing behavior is unchanged until this is explicitly
    # switched on. See README's Tools section / Phase 3.2 notes.
    business_service_provider: str = "fake"

    # Required only when business_service_provider == "strapi". Strapi
    # (admin/) is the real business backend for Car/Booking/Payment data
    # (see Phase 3.1's audit) - never touched by this service directly,
    # only through its authenticated REST API.
    strapi_url: str = "http://localhost:1337"

    # No default on purpose, same reasoning as openai_api_key: a missing
    # token must fail loudly rather than silently falling back. Strapi's
    # Public role permissions for Car/Booking/Payment are intentionally
    # left disabled (see Phase 3.1's audit) - this service authenticates
    # with a scoped, read-only API token instead, created manually in
    # Strapi Admin (Settings > API Tokens). Never log or print this value.
    strapi_api_token: Optional[str] = None

    # Selects which RagAnswerer implementation backs ChatService (see
    # app/chats/service.py:get_rag_answerer). One of: "existing"
    # (AgentService - retrieval + bounded tool-calling + LLMProvider,
    # unchanged since Step 15), "langchain" (Phase 2.8:
    # KnowledgeBaseRetriever + LangChainRagGenerationService). Defaults
    # to "existing" - the LangChain path is additive and opt-in, not a
    # replacement, until there's a reason to change the default.
    rag_provider: str = "existing"


@lru_cache
def get_settings() -> Settings:
    """Cached Settings accessor, used as a FastAPI dependency via Depends()."""
    return Settings()
