"""Minimal LangChain RAG generation: composes KnowledgeBaseRetriever
(retriever.py) with a LangChain chat model to produce a grounded
answer, plus the sources that supported it.

Not an agent: one retrieval call, one prompt, one chat model call per
question - no tools, no loop, no LangGraph. Mirrors this project's
existing AnswerGenerator (app/rag/answer_generator.py) - same shape
(Retriever + chat model -> grounded answer with sources), same
"return a fixed not-available answer instead of calling the LLM with no
context" rule - just built from LangChain primitives instead of this
project's own LLMProvider Protocol, and kept in this package rather
than app/rag because it depends on langchain_core/langchain_openai.
"""
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core.config import Settings
from app.langchain_integration.retriever import KnowledgeBaseRetriever, get_langchain_retriever

NOT_AVAILABLE_ANSWER = "I don't have information about that in the current knowledge base."

SYSTEM_INSTRUCTIONS = (
    "You are a car rental assistant. Answer the question using only the knowledge "
    "context below - it is your source of truth. Do not invent car rental policies, "
    "prices, requirements, or availability that are not stated in the context. "
    "Preserve any numbers, percentages, dates, or conditions from the context exactly "
    "as written - never round, approximate, or paraphrase a policy figure. If the "
    "context does not contain enough information to answer, say so plainly instead "
    "of guessing."
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS),
        ("human", "Knowledge context:\n{context}\n\nQuestion: {question}"),
    ]
)


class RagSource(BaseModel):
    """Minimal citation, mirroring app.rag.models.AnswerSource's shape
    and reasoning (enough to say which document/section supported an
    answer, nothing more) - kept as its own type rather than imported
    directly, so this LangChain-specific boundary and app/rag (which
    has no LangChain dependency) stay free to evolve independently.
    """

    document_title: str
    section_heading: Optional[str] = None


class RagAnswer(BaseModel):
    answer: str
    sources: List[RagSource]


class LangChainRagGenerationService:
    """Retriever + chat model -> grounded answer. Deliberately a
    separate class from KnowledgeBaseRetriever: this class owns
    prompting and generation, the retriever owns retrieval - one class,
    one responsibility, the same split AnswerGenerator/VectorRetriever
    already establish elsewhere in this project.
    """

    def __init__(self, retriever: KnowledgeBaseRetriever, chat_model: BaseChatModel) -> None:
        self._retriever = retriever
        self._chat_model = chat_model
        self._chain = _PROMPT | self._chat_model

    async def agenerate(self, question: str) -> RagAnswer:
        documents = await self._retriever.ainvoke(question)

        if not documents:
            # Never call the LLM with no context and hope it declines to
            # guess - the application layer, not the model, is the
            # source of truth for "we have nothing relevant" (same rule
            # AnswerGenerator.answer() already applies).
            return RagAnswer(answer=NOT_AVAILABLE_ANSWER, sources=[])

        context = _build_context(documents)
        response = await self._chain.ainvoke({"context": context, "question": question})

        return RagAnswer(
            answer=response.content,
            sources=[_to_source(document) for document in documents],
        )


def _build_context(documents: List[Document]) -> str:
    """Turns retrieved Documents into the text block the model reads as
    its knowledge context. Only document_title, section_heading, and
    page_content are included - chunk_id/position/score mean nothing to
    the model and would just be noise in the prompt (mirrors
    app.rag.answer_generator.build_context's same reasoning).
    """
    blocks = []
    for document in documents:
        heading = document.metadata.get("section_heading")
        heading_line = f" — {heading}" if heading else ""
        blocks.append(f"Source: {document.metadata['document_title']}{heading_line}\n{document.page_content}")
    return "\n\n".join(blocks)


def _to_source(document: Document) -> RagSource:
    return RagSource(
        document_title=document.metadata["document_title"],
        section_heading=document.metadata.get("section_heading"),
    )


def get_rag_generation_service(settings: Settings) -> LangChainRagGenerationService:
    """Composition point, matching every other module's plain-function
    pattern (get_langchain_retriever, get_retriever, get_llm_provider).
    The chat model name is never hard-coded: it comes from
    Settings.openai_model, the same setting app/llm/openai_provider.py
    already uses for chat - one setting decides which OpenAI chat model
    this project uses, regardless of which orchestration layer calls it.
    temperature=0 favors literal, low-variance answers over creative
    ones - appropriate for a system whose whole job is repeating policy
    figures exactly, not writing varied prose.
    """
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required to build the LangChain RAG generation service"
        )

    chat_model = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    return LangChainRagGenerationService(
        retriever=get_langchain_retriever(settings),
        chat_model=chat_model,
    )
