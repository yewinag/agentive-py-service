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
    # One of: "memory", "pgvector".
    vector_store_provider: str = "memory"

    # Required only when vector_store_provider == "pgvector". No default
    # on purpose, same reasoning as openai_api_key. Expected form:
    # postgresql+asyncpg://user:password@host:5432/dbname
    database_url: Optional[str] = None

    # Default VectorRetriever.retrieve() top_k, overridable per call.
    retrieval_top_k: int = 5

    # No default on purpose (None = no filtering): there is no empirical
    # basis yet for a universal similarity cutoff for a given embedding
    # model/knowledge base. See README's Retrieval section.
    retrieval_min_score: Optional[float] = None


@lru_cache
def get_settings() -> Settings:
    """Cached Settings accessor, used as a FastAPI dependency via Depends()."""
    return Settings()
