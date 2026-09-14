from app.llm.dependencies import get_llm_provider
from app.main import app


def test_chat_returns_reply_from_llm_provider(client):
    response = client.post("/api/v1/chat", json={"message": "Hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "[fake-llm-reply] Hello"


def test_chat_rejects_blank_message(client):
    response = client.post("/api/v1/chat", json={"message": "   "})

    assert response.status_code == 422


def test_chat_router_does_not_require_a_specific_provider_implementation(client):
    class StubProvider:
        """Satisfies LLMProvider structurally, unrelated to FakeLLMProvider."""

        async def generate_reply(self, message: str) -> str:
            return f"stub:{message}"

    app.dependency_overrides[get_llm_provider] = lambda: StubProvider()
    try:
        response = client.post("/api/v1/chat", json={"message": "Hi"})
    finally:
        app.dependency_overrides.pop(get_llm_provider, None)

    assert response.status_code == 200
    assert response.json()["reply"] == "stub:Hi"
