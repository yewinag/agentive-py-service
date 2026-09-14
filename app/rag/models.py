from typing import Optional

from pydantic import BaseModel


class AnswerSource(BaseModel):
    """Minimal citation: enough to say which document/section supported
    an answer, without building a full citation framework yet. Excludes
    chunk text, id, position, and score - none of that is meaningful to
    a caller deciding whether to trust an answer.
    """

    document_title: str
    section_heading: Optional[str] = None


class GroundedAnswer(BaseModel):
    """AnswerGenerator's result. Kept internal to app/rag for now rather
    than folded into ChatResponse - see README's Retrieval/RAG section
    for why exposing sources over /api/v1/chat is deferred, not dropped.
    """

    answer: str
    sources: list[AnswerSource]
