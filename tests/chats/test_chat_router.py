from app.chats.service import get_chat_service
from app.main import app
from app.rag.models import AnswerSource, GroundedAnswer


class StubChatService:
    """Satisfies the shape ChatService/get_reply() offers, unrelated to
    the real ChatService/AnswerGenerator chain - proves the router
    depends on whatever get_chat_service() provides via Depends(), not
    a hardcoded retrieval/RAG pipeline.
    """

    def __init__(self, result: GroundedAnswer) -> None:
        self._result = result

    async def get_reply(self, message: str) -> GroundedAnswer:
        return self._result


def _override_chat_service(result: GroundedAnswer):
    app.dependency_overrides[get_chat_service] = lambda: StubChatService(result)


def test_chat_returns_grounded_reply_and_sources(client):
    _override_chat_service(
        GroundedAnswer(
            answer="You must be at least 21 years old.",
            sources=[
                AnswerSource(
                    document_title="Terms & Rental Policies",
                    section_heading="1. Driver Eligibility & Required Documents",
                )
            ],
        )
    )
    try:
        response = client.post("/api/v1/chat", json={"message": "What is the minimum age?"})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "You must be at least 21 years old."
    assert body["sources"] == [
        {
            "document_title": "Terms & Rental Policies",
            "section_heading": "1. Driver Eligibility & Required Documents",
        }
    ]


def test_chat_returns_empty_sources_for_not_available_answer(client):
    _override_chat_service(
        GroundedAnswer(
            answer="I don't have information about that in the current knowledge base.",
            sources=[],
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


def test_chat_router_does_not_hardcode_a_specific_chat_service_implementation(client):
    _override_chat_service(GroundedAnswer(answer="stubbed answer", sources=[]))
    try:
        response = client.post("/api/v1/chat", json={"message": "Hi"})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 200
    assert response.json()["reply"] == "stubbed answer"
