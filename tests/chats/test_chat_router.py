from app.chats.service import ChatResult, get_chat_service
from app.main import app
from app.rag.models import AnswerSource, GroundedAnswer


class StubChatService:
    """Satisfies the shape ChatService/get_reply() offers, unrelated to
    the real ChatService/AnswerGenerator/ConversationStore chain - proves
    the router depends on whatever get_chat_service() provides via
    Depends(), not a hardcoded retrieval/RAG/conversation pipeline.
    """

    def __init__(self, result: ChatResult) -> None:
        self._result = result
        self.calls = []

    async def get_reply(self, message: str, conversation_id=None) -> ChatResult:
        self.calls.append((message, conversation_id))
        return self._result


def _override_chat_service(result: ChatResult) -> StubChatService:
    stub = StubChatService(result)
    app.dependency_overrides[get_chat_service] = lambda: stub
    return stub


def test_chat_returns_grounded_reply_sources_and_conversation_id(client):
    _override_chat_service(
        ChatResult(
            conversation_id="conv-1",
            answer=GroundedAnswer(
                answer="You must be at least 21 years old.",
                sources=[
                    AnswerSource(
                        document_title="Terms & Rental Policies",
                        section_heading="1. Driver Eligibility & Required Documents",
                    )
                ],
            ),
        )
    )
    try:
        response = client.post("/api/v1/chat", json={"message": "What is the minimum age?"})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "You must be at least 21 years old."
    assert body["conversation_id"] == "conv-1"
    assert body["sources"] == [
        {
            "document_title": "Terms & Rental Policies",
            "section_heading": "1. Driver Eligibility & Required Documents",
        }
    ]


def test_chat_returns_empty_sources_for_not_available_answer(client):
    _override_chat_service(
        ChatResult(
            conversation_id="conv-2",
            answer=GroundedAnswer(
                answer="I don't have information about that in the current knowledge base.",
                sources=[],
            ),
        )
    )
    try:
        response = client.post("/api/v1/chat", json={"message": "What's the weather?"})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_chat_rejects_blank_message(client):
    response = client.post("/api/v1/chat", json={"message": "   "})

    assert response.status_code == 422


def test_chat_works_without_conversation_id_for_backward_compatibility(client):
    stub = _override_chat_service(
        ChatResult(conversation_id="new-conv", answer=GroundedAnswer(answer="stubbed answer", sources=[]))
    )
    try:
        response = client.post("/api/v1/chat", json={"message": "Hi"})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 200
    assert response.json()["reply"] == "stubbed answer"
    assert stub.calls == [("Hi", None)]


def test_chat_forwards_supplied_conversation_id_to_chat_service(client):
    stub = _override_chat_service(
        ChatResult(conversation_id="conv-3", answer=GroundedAnswer(answer="follow-up answer", sources=[]))
    )
    try:
        response = client.post(
            "/api/v1/chat", json={"message": "And what about deposits?", "conversation_id": "conv-3"}
        )
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 200
    assert stub.calls == [("And what about deposits?", "conv-3")]
