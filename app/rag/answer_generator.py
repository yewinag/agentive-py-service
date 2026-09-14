from app.core.config import Settings
from app.knowledge.retriever import Retriever
from app.knowledge.vector_store import VectorSearchResult
from app.llm.provider import LLMProvider
from app.rag.models import AnswerSource, GroundedAnswer

NOT_AVAILABLE_ANSWER = (
    "I don't have information about that in the current knowledge base."
)

GROUNDING_INSTRUCTIONS = (
    "You are a car rental assistant. Answer the question using only the "
    "knowledge context below - it is your source of truth. Do not invent "
    "car rental policies, prices, requirements, or availability that are "
    "not stated in the context. Clearly distinguish information the "
    "context supports from anything you are not certain of. If the "
    "context does not contain enough information to answer, say so "
    "instead of guessing."
)


class AnswerGenerator:
    """Orchestrates retrieval + grounded answer generation. Depends only
    on the Retriever and LLMProvider Protocols - never a concrete
    OpenAI/pgvector/FastAPI type.

    No Protocol of its own, deliberately: like ChatService, this is
    application orchestration logic with exactly one implementation, not
    an external boundary with swappable backends - testability already
    comes from Retriever and LLMProvider each being fakeable.
    """

    def __init__(self, retriever: Retriever, llm_provider: LLMProvider) -> None:
        self._retriever = retriever
        self._llm_provider = llm_provider

    async def answer(self, question: str) -> GroundedAnswer:
        results = await self._retriever.retrieve(question)

        if not results:
            # Never call the LLM with no context and hope it declines to
            # guess - the application layer, not the model, is the
            # source of truth for "we have nothing relevant."
            return GroundedAnswer(answer=NOT_AVAILABLE_ANSWER, sources=[])

        prompt = build_prompt(question, build_context(results))
        answer_text = await self._llm_provider.generate_reply(prompt)

        return GroundedAnswer(
            answer=answer_text,
            sources=[_to_source(result) for result in results],
        )


def build_context(results: list[VectorSearchResult]) -> str:
    """Turns ranked retrieval results into the text block the LLM reads
    as its knowledge context. Deterministic: the same results always
    produce the same string. Only document_title, section_heading, and
    text are included - score/id/position mean nothing to the model and
    would just be noise in the prompt.
    """
    blocks = []
    for result in results:
        chunk = result.chunk
        heading_line = f" — {chunk.section_heading}" if chunk.section_heading else ""
        blocks.append(f"Source: {chunk.document_title}{heading_line}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, context: str) -> str:
    return (
        f"{GROUNDING_INSTRUCTIONS}\n\n"
        f"Knowledge context:\n{context}\n\n"
        f"Question: {question}"
    )


def _to_source(result: VectorSearchResult) -> AnswerSource:
    return AnswerSource(
        document_title=result.chunk.document_title,
        section_heading=result.chunk.section_heading,
    )


def get_answer_generator(settings: Settings) -> AnswerGenerator:
    """Composition point - a plain function, not FastAPI Depends()-wired,
    for the same reason as get_retriever(): no endpoint consumes this
    yet. Builds its Retriever and LLMProvider via their own existing
    composition points rather than deciding providers itself.
    """
    from app.knowledge.retriever import get_retriever
    from app.llm.dependencies import get_llm_provider

    return AnswerGenerator(
        retriever=get_retriever(settings),
        llm_provider=get_llm_provider(settings),
    )
