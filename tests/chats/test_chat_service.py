import asyncio

from app.chats.service import ChatService
from app.conversation.in_memory_store import InMemoryConversationStore
from app.rag.models import GroundedAnswer


class StubAnswerGenerator:
    """Records exactly what (question, history) ChatService passed to
    AnswerGenerator.answer(), without depending on a real Retriever/
    LLMProvider (already covered by Step 11's tests).
    """

    def __init__(self, reply: str = "an answer") -> None:
        self.calls = []
        self.reply = reply

    async def answer(self, question, history=None):
        self.calls.append((question, list(history) if history else []))
        return GroundedAnswer(answer=self.reply, sources=[])


def test_new_chat_without_conversation_id_creates_a_conversation():
    service = ChatService(StubAnswerGenerator(), InMemoryConversationStore(), history_window=6)

    result = asyncio.run(service.get_reply("Hello"))

    assert result.conversation_id
    assert result.answer.answer == "an answer"


def test_followup_with_conversation_id_reuses_the_same_conversation():
    store = InMemoryConversationStore()
    service = ChatService(StubAnswerGenerator(), store, history_window=6)

    first = asyncio.run(service.get_reply("Hello"))
    second = asyncio.run(service.get_reply("Follow-up", conversation_id=first.conversation_id))

    assert second.conversation_id == first.conversation_id


def test_conversation_is_updated_with_user_message_and_assistant_response():
    store = InMemoryConversationStore()
    service = ChatService(StubAnswerGenerator(reply="the answer"), store, history_window=6)

    result = asyncio.run(service.get_reply("What is the deposit?"))
    messages = asyncio.run(store.get_recent_messages(result.conversation_id, limit=10))

    assert [(m.role, m.content) for m in messages] == [
        ("user", "What is the deposit?"),
        ("assistant", "the answer"),
    ]


def test_followup_request_passes_previous_turns_as_history_to_answer_generator():
    store = InMemoryConversationStore()
    generator = StubAnswerGenerator(reply="first answer")
    service = ChatService(generator, store, history_window=6)

    first = asyncio.run(service.get_reply("First question"))
    generator.reply = "second answer"
    asyncio.run(service.get_reply("Second question", conversation_id=first.conversation_id))

    first_question, first_history = generator.calls[0]
    second_question, second_history = generator.calls[1]
    assert first_history == []
    assert second_question == "Second question"
    assert [m.content for m in second_history] == ["First question", "first answer"]


def test_unknown_conversation_id_falls_back_to_a_new_conversation_instead_of_erroring():
    service = ChatService(StubAnswerGenerator(), InMemoryConversationStore(), history_window=6)

    result = asyncio.run(service.get_reply("Hello", conversation_id="does-not-exist"))

    assert result.conversation_id != "does-not-exist"


def test_history_window_bounds_how_many_previous_messages_are_sent():
    store = InMemoryConversationStore()
    generator = StubAnswerGenerator()
    service = ChatService(generator, store, history_window=2)

    conversation_id = None
    for i in range(3):
        result = asyncio.run(service.get_reply(f"question {i}", conversation_id=conversation_id))
        conversation_id = result.conversation_id

    _, last_history = generator.calls[-1]
    assert len(last_history) <= 2
