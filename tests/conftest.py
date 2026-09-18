import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app


@pytest.fixture(autouse=True, scope="session")
def _isolate_settings_from_the_real_env_file():
    """Settings(...) reads pydantic-settings' normal precedence: init
    kwargs > real environment variables > the .env file > class
    defaults. Tests construct Settings with only the fields a given
    test cares about (e.g. Settings(vector_store_provider="memory")),
    relying on every other field falling back to its class default
    (e.g. embedding_provider="fake") - but a developer's real .env file
    sits ABOVE that default in precedence. A real .env configured for
    actual local development (e.g. EMBEDDING_PROVIDER=openai plus a
    real OPENAI_API_KEY, needed for the real Qdrant ingestion command)
    would otherwise silently leak into test Settings construction,
    making "unit" tests issue real, billed OpenAI API calls. Tests must
    be hermetic regardless of what's in the developer's own .env, so
    .env loading is disabled for the whole test session; any test that
    genuinely wants to assert something about a specific env var still
    sets it explicitly via monkeypatch.setenv or a constructor kwarg.
    """
    Settings.model_config["env_file"] = None


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
