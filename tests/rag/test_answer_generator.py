import asyncio

import pytest

from app.conversation.models import ConversationMessage
from app.knowledge.exceptions import EmbeddingProviderError, VectorStoreError
from app.knowledge.models import DocumentChunk
from app.knowledge.vector_store import VectorSearchResult
from app.llm.exceptions import LLMProviderError
from app.rag.answer_generator import (
    NOT_AVAILABLE_ANSWER,
    AnswerGenerator,
    build_context,
    build_history_block,
    build_prompt,
)


class StubRetriever:
    def __init__(self, results=None, error=None):
        self.calls = []
        self._results = results if results is not None else []
        self._error = error

    async def retrieve(self, query, top_k=None, min_score=None):
        self.calls.append(query)
        if self._error:
            raise self._error
        return self._results


class StubLLMProvider:
    def __init__(self, reply="a grounded answer", error=None):
        self.received_prompts = []
        self._reply = reply
        self._error = error

    async def generate_reply(self, message):
        self.received_prompts.append(message)
        if self._error:
            raise self._error
        return self._reply


def _result(document_title: str, section_heading, text: str, score: float = 0.9) -> VectorSearchResult:
    return VectorSearchResult(
        chunk=DocumentChunk(
            id=f"{document_title}-{section_heading}",
            document_id="doc-1",
            document_title=document_title,
            section_heading=section_heading,
            text=text,
            position=0,
        ),
        score=score,
    )


def test_build_context_includes_title_heading_and_text():
    results = [
        _result("Terms & Rental Policies", "1. Driver Eligibility", "Minimum Age: 21."),
    ]

    context = build_context(results)

    assert "Terms & Rental Policies" in context
    assert "1. Driver Eligibility" in context
    assert "Minimum Age: 21." in context


def test_build_context_omits_heading_line_when_none():
    results = [_result("Doc", None, "Some preamble text.")]

    context = build_context(results)

    assert context == "Source: Doc\nSome preamble text."


def test_build_context_is_deterministic():
    results = [
        _result("Doc A", "1. Section", "text a"),
        _result("Doc B", "2. Section", "text b"),
    ]

    assert build_context(results) == build_context(results)


def test_build_prompt_includes_grounding_instructions_context_and_question():
    prompt = build_prompt("What is the minimum age?", "Source: Doc\nMinimum Age: 21.")

    assert "Minimum Age: 21." in prompt
    assert "What is the minimum age?" in prompt
    assert "do not invent" in prompt.lower()


def test_answer_sends_grounded_prompt_to_llm_when_results_found():
    llm = StubLLMProvider(reply="You must be 21.")
    retriever = StubRetriever(results=[_result("Policies", "1. Eligibility", "Minimum Age: 21.")])
    generator = AnswerGenerator(retriever, llm)

    result = asyncio.run(generator.answer("What is the minimum age?"))

    assert result.answer == "You must be 21."
    assert "Minimum Age: 21." in llm.received_prompts[0]
    assert "What is the minimum age?" in llm.received_prompts[0]


def test_answer_includes_sources_for_the_retrieved_chunks():
    retriever = StubRetriever(
        results=[_result("Policies", "1. Eligibility", "Minimum Age: 21.")]
    )
    generator = AnswerGenerator(retriever, StubLLMProvider())

    result = asyncio.run(generator.answer("question"))

    assert len(result.sources) == 1
    assert result.sources[0].document_title == "Policies"
    assert result.sources[0].section_heading == "1. Eligibility"


def test_answer_does_not_call_llm_when_retrieval_returns_no_results():
    llm = StubLLMProvider()
    retriever = StubRetriever(results=[])
    generator = AnswerGenerator(retriever, llm)

    result = asyncio.run(generator.answer("something unrelated"))

    assert result.answer == NOT_AVAILABLE_ANSWER
    assert result.sources == []
    assert llm.received_prompts == []


def test_answer_propagates_retriever_errors():
    retriever = StubRetriever(error=EmbeddingProviderError("boom"))
    generator = AnswerGenerator(retriever, StubLLMProvider())

    with pytest.raises(EmbeddingProviderError):
        asyncio.run(generator.answer("question"))


def test_answer_propagates_vector_store_errors_from_retriever():
    retriever = StubRetriever(error=VectorStoreError("boom"))
    generator = AnswerGenerator(retriever, StubLLMProvider())

    with pytest.raises(VectorStoreError):
        asyncio.run(generator.answer("question"))


def test_answer_propagates_llm_provider_errors():
    retriever = StubRetriever(results=[_result("Doc", "1. Section", "text")])
    llm = StubLLMProvider(error=LLMProviderError("boom"))
    generator = AnswerGenerator(retriever, llm)

    with pytest.raises(LLMProviderError):
        asyncio.run(generator.answer("question"))


def test_build_history_block_is_empty_for_no_history():
    assert build_history_block(None) == ""
    assert build_history_block([]) == ""


def test_build_history_block_formats_roles_and_content_in_order():
    history = [
        ConversationMessage(role="user", content="What is the deposit?"),
        ConversationMessage(role="assistant", content="It is 5,000 THB."),
    ]

    block = build_history_block(history)

    assert block == "User: What is the deposit?\nAssistant: It is 5,000 THB."


def test_build_prompt_omits_history_section_when_no_history():
    prompt = build_prompt("question", "context")

    assert "Conversation so far" not in prompt


def test_build_prompt_includes_history_section_when_history_given():
    history = [ConversationMessage(role="user", content="Earlier question")]

    prompt = build_prompt("question", "context", history)

    assert "Conversation so far:" in prompt
    assert "User: Earlier question" in prompt


def test_answer_passes_history_into_the_prompt_sent_to_the_llm():
    llm = StubLLMProvider()
    retriever = StubRetriever(results=[_result("Policies", "1. Eligibility", "Minimum Age: 21.")])
    generator = AnswerGenerator(retriever, llm)
    history = [
        ConversationMessage(role="user", content="What documents do I need?"),
        ConversationMessage(role="assistant", content="A driver's license and ID."),
    ]

    asyncio.run(generator.answer("And what about payment?", history=history))

    prompt = llm.received_prompts[0]
    assert "Conversation so far:" in prompt
    assert "User: What documents do I need?" in prompt
    assert "Assistant: A driver's license and ID." in prompt


def test_retrieve_is_called_with_only_the_current_question_not_history():
    """Retrieval must stay focused on the current information need - see
    README's RAG interaction note. history must never reach Retriever.
    """
    retriever = StubRetriever(results=[_result("Doc", "1. Section", "text")])
    generator = AnswerGenerator(retriever, StubLLMProvider())
    history = [ConversationMessage(role="user", content="unrelated earlier question")]

    asyncio.run(generator.answer("current question", history=history))

    assert retriever.calls == ["current question"]
