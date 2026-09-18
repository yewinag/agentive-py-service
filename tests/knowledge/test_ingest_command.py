"""Unit-level coverage for the explicit ingestion command
(app/knowledge/ingest.py) that needs neither a real Qdrant instance nor
real PDFs - just the command's own control flow. The real, full
PDF-to-Qdrant round trip lives in
test_qdrant_ingestion_command_integration.py (skipped unless a real
Qdrant is available).
"""
import asyncio

import pytest

from app.core.config import Settings
from app.knowledge.ingest import run


def test_run_refuses_when_vector_store_provider_is_not_qdrant(capsys):
    settings = Settings(vector_store_provider="memory")

    exit_code = asyncio.run(run(settings, reset=False))

    assert exit_code == 1
    assert "not 'qdrant'" in capsys.readouterr().err


@pytest.mark.parametrize("provider", ["pgvector", "memory"])
def test_run_refuses_for_every_non_qdrant_provider(provider, capsys):
    settings = Settings(vector_store_provider=provider)

    exit_code = asyncio.run(run(settings, reset=False))

    assert exit_code == 1


def test_run_fails_cleanly_when_openai_embedding_provider_has_no_api_key(capsys):
    """Must report a clean configuration error (Phase 2.7.2's explicit
    "stop before making API calls and report exactly what configuration
    is missing" requirement) rather than an unhandled traceback partway
    through - no embedding or Qdrant call should ever be attempted.
    """
    settings = Settings(
        vector_store_provider="qdrant",
        embedding_provider="openai",
        openai_api_key=None,
    )

    exit_code = asyncio.run(run(settings, reset=False))

    output = capsys.readouterr()
    assert exit_code == 1
    assert "OPENAI_API_KEY is required" in output.err
    assert "Status: FAILED" in output.out
