from functools import lru_cache

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
    # Only "fake" exists today; a real provider adds another accepted value.
    llm_provider: str = "fake"


@lru_cache
def get_settings() -> Settings:
    """Cached Settings accessor, used as a FastAPI dependency via Depends()."""
    return Settings()
