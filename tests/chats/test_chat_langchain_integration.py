"""Real, cost-controlled integration coverage proving RAG_PROVIDER=langchain
actually wires KnowledgeBaseRetriever + LangChainRagGenerationService
into the real POST /api/v1/chat endpoint - real Qdrant, real OpenAI
embeddings and chat completion, no fakes, no simulated results.

COST AWARENESS: exactly 4 real chat-completion calls total (one per
test function) plus their retrieval's query embeddings - no loops, no
parametrization over a larger matrix. See
tests/langchain_integration/test_generation_integration.py, which
already covers the underlying generation service directly; this file's
job is only to prove the HTTP wiring (RAG_PROVIDER -> get_rag_answerer
-> ChatService -> the real endpoint), not to re-derive semantic
correctness from scratch.

Skipped unless both QDRANT_URL and OPENAI_API_KEY are set as real
environment variables (not read from .env - see conftest.py). Never
runs in the default `pytest -v` invocation. To run it locally:

    docker compose up -d qdrant
    QDRANT_URL=http://localhost:6333 OPENAI_API_KEY=sk-... \\
        python -m pytest tests/chats/test_chat_langchain_integration.py -v

Read-only against the knowledge base: no ingestion, no reset.
"""
import os

import pytest

from app.chats.service import get_chat_service
from app.core.config import Settings
from app.main import app

QDRANT_URL = os.environ.get("QDRANT_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

pytestmark = pytest.mark.skipif(
    not (QDRANT_URL and OPENAI_API_KEY),
    reason="QDRANT_URL and/or OPENAI_API_KEY not set - skipping real langchain-mode chat integration test",
)


def _langchain_settings() -> Settings:
    return Settings(
        rag_provider="langchain",
        vector_store_provider="qdrant",
        embedding_provider="openai",
        qdrant_url=QDRANT_URL,
        qdrant_collection="knowledge_chunk_embeddings",
        openai_api_key=OPENAI_API_KEY,
    )


def _post_chat(client, message: str):
    chat_service = get_chat_service(_langchain_settings())
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    try:
        return client.post("/api/v1/chat", json={"message": message})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)


def test_cancellation_question_via_langchain_mode(client):
    response = _post_chat(client, "What happens if I cancel less than 24 hours before pickup?")

    assert response.status_code == 200
    body = response.json()
    assert "50%" in body["reply"]
    assert body["conversation_id"]
    assert any(source["document_title"] == "Cancellation Policy" for source in body["sources"])


def test_security_deposit_question_via_langchain_mode(client):
    response = _post_chat(client, "How much is the security deposit?")

    assert response.status_code == 200
    body = response.json()
    normalized = body["reply"].replace(",", "")
    assert "5000" in normalized
    assert "10000" in normalized
    assert any(source["document_title"] == "Payment Policy" for source in body["sources"])


def test_minimum_age_question_via_langchain_mode(client):
    response = _post_chat(client, "What is the minimum age to rent a car?")

    assert response.status_code == 200
    body = response.json()
    assert "21" in body["reply"]


def test_unsupported_question_via_langchain_mode_does_not_invent_a_policy(client):
    response = _post_chat(client, "Do you offer free baby seats?")

    assert response.status_code == 200
    body = response.json()
    # The model must not confidently invent a "yes" (or a specific policy)
    # for something the knowledge base never mentions - same real-model
    # grounding check as test_generation_integration.py's equivalent test.
    lowered = body["reply"].lower()
    uncertainty_phrases = [
        "don't have", "do not have", "no information", "not mention",
        "doesn't mention", "not stated", "not available", "does not provide",
        "doesn't provide", "cannot answer", "can't answer", "does not contain",
        "doesn't contain", "not specify", "does not specify",
    ]
    assert any(phrase in lowered for phrase in uncertainty_phrases), (
        f"Expected the model to decline rather than invent an answer, got: {body['reply']!r}"
    )
