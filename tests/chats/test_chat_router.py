def test_chat_returns_stub_reply(client):
    response = client.post("/api/v1/chat", json={"message": "Hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "You said: Hello"


def test_chat_rejects_blank_message(client):
    response = client.post("/api/v1/chat", json={"message": "   "})

    assert response.status_code == 422
