from fastapi import Request
from fastapi.responses import JSONResponse

from app.knowledge.exceptions import EmbeddingProviderError, VectorStoreError
from app.llm.exceptions import LLMProviderError

DEPENDENCY_FAILURE_EXCEPTIONS = (EmbeddingProviderError, VectorStoreError, LLMProviderError)


async def handle_dependency_failure(request: Request, exc: Exception) -> JSONResponse:
    """Maps a known, already-translated provider/storage failure
    (EmbeddingProviderError, VectorStoreError, LLMProviderError) to a
    clean, generic 503 response. These exceptions exist specifically so
    a raw SDK/database error never reaches this point; this handler's
    job is only to give the client a stable, actionable status instead
    of FastAPI's default 500, and to keep the message identical
    regardless of which dependency actually failed - never echoing
    exc's own message, which could contain provider-specific detail.
    """
    return JSONResponse(
        status_code=503,
        content={"detail": "The chat service is temporarily unavailable. Please try again."},
    )
