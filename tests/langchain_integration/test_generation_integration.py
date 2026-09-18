"""Real, cost-controlled integration coverage for
LangChainRagGenerationService against the real Qdrant collection, real
OpenAI embeddings, and a real OpenAI chat completion.

COST AWARENESS: this file makes exactly 4 real chat-completion calls in
total (one per test function below) plus their retrieval's query
embeddings - no loops, no parametrization over a larger matrix, no
retries. Do not add more real-call test functions to this file without
reconsidering the cost tradeoff.

Skipped unless both QDRANT_URL and OPENAI_API_KEY are set as real
environment variables (not read from .env - see conftest.py). Never
runs in the default `pytest -v` invocation. To run it locally:

    docker compose up -d qdrant
    QDRANT_URL=http://localhost:6333 OPENAI_API_KEY=sk-... \\
        python -m pytest tests/langchain_integration/test_generation_integration.py -v

Read-only against the knowledge base: no ingestion, no reset.
"""
import asyncio
import os

import pytest

from app.core.config import Settings
from app.langchain_integration.generation import NOT_AVAILABLE_ANSWER, get_rag_generation_service

QDRANT_URL = os.environ.get("QDRANT_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

pytestmark = pytest.mark.skipif(
    not (QDRANT_URL and OPENAI_API_KEY),
    reason="QDRANT_URL and/or OPENAI_API_KEY not set - skipping real Qdrant+OpenAI generation test",
)


def _settings() -> Settings:
    return Settings(
        vector_store_provider="qdrant",
        embedding_provider="openai",
        qdrant_url=QDRANT_URL,
        qdrant_collection="knowledge_chunk_embeddings",
        openai_api_key=OPENAI_API_KEY,
    )


def _contains_uncertainty_language(answer: str) -> bool:
    lowered = answer.lower()
    phrases = [
        "don't have", "do not have", "no information", "not mention",
        "doesn't mention", "not stated", "not available", "not specify",
        "does not specify", "cannot find", "can't find", "not provided",
        "not include", "doesn't include", "not covered", "does not provide",
        "doesn't provide", "cannot answer", "can't answer", "unable to answer",
        "does not contain", "doesn't contain",
    ]
    return any(phrase in lowered for phrase in phrases)


def test_cancellation_question_states_the_50_percent_fee():
    service = get_rag_generation_service(_settings())

    result = asyncio.run(service.agenerate("What happens if I cancel less than 24 hours before pickup?"))

    assert "50%" in result.answer
    assert result.answer != NOT_AVAILABLE_ANSWER
    assert any(source.document_title == "Cancellation Policy" for source in result.sources)


def test_security_deposit_question_states_the_deposit_range():
    service = get_rag_generation_service(_settings())

    result = asyncio.run(service.agenerate("How much is the security deposit?"))

    normalized = result.answer.replace(",", "")
    assert "5000" in normalized
    assert "10000" in normalized
    assert result.answer != NOT_AVAILABLE_ANSWER
    assert any(source.document_title == "Payment Policy" for source in result.sources)


def test_minimum_age_question_states_21():
    service = get_rag_generation_service(_settings())

    result = asyncio.run(service.agenerate("What is the minimum age to rent a car?"))

    assert "21" in result.answer
    assert result.answer != NOT_AVAILABLE_ANSWER


def test_unsupported_question_does_not_invent_an_answer():
    service = get_rag_generation_service(_settings())

    result = asyncio.run(service.agenerate("Do you offer free baby seats?"))

    # The retriever will still return its nearest (irrelevant) chunks -
    # Qdrant always has neighbors - so the model itself, not the
    # short-circuit, must recognize the context doesn't support an
    # answer. Real LLM wording varies even at temperature=0, so this
    # checks for uncertainty language rather than one exact sentence.
    assert result.answer == NOT_AVAILABLE_ANSWER or _contains_uncertainty_language(result.answer), (
        f"Expected the model to decline rather than invent an answer, got: {result.answer!r}"
    )
