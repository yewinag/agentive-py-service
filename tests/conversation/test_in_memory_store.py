import asyncio

import pytest

from app.conversation.exceptions import ConversationNotFoundError
from app.conversation.in_memory_store import InMemoryConversationStore
from app.conversation.models import ConversationMessage


def _msg(role: str, content: str) -> ConversationMessage:
    return ConversationMessage(role=role, content=content)


def test_create_returns_a_new_empty_conversation_with_a_fresh_id():
    store = InMemoryConversationStore()

    conversation = asyncio.run(store.create())

    assert conversation.id
    assert conversation.messages == []


def test_create_returns_distinct_ids_across_calls():
    store = InMemoryConversationStore()

    first = asyncio.run(store.create())
    second = asyncio.run(store.create())

    assert first.id != second.id


def test_get_returns_the_created_conversation():
    store = InMemoryConversationStore()
    created = asyncio.run(store.create())

    fetched = asyncio.run(store.get(created.id))

    assert fetched is not None
    assert fetched.id == created.id


def test_get_returns_none_for_an_unknown_conversation_id():
    store = InMemoryConversationStore()

    result = asyncio.run(store.get("does-not-exist"))

    assert result is None


def test_append_message_preserves_order():
    store = InMemoryConversationStore()
    conversation = asyncio.run(store.create())

    async def scenario():
        await store.append_message(conversation.id, _msg("user", "first"))
        await store.append_message(conversation.id, _msg("assistant", "second"))
        await store.append_message(conversation.id, _msg("user", "third"))
        return await store.get_recent_messages(conversation.id, limit=10)

    messages = asyncio.run(scenario())

    assert [m.content for m in messages] == ["first", "second", "third"]
    assert [m.role for m in messages] == ["user", "assistant", "user"]


def test_append_message_raises_for_an_unknown_conversation():
    store = InMemoryConversationStore()

    with pytest.raises(ConversationNotFoundError):
        asyncio.run(store.append_message("does-not-exist", _msg("user", "hi")))


def test_get_recent_messages_raises_for_an_unknown_conversation():
    store = InMemoryConversationStore()

    with pytest.raises(ConversationNotFoundError):
        asyncio.run(store.get_recent_messages("does-not-exist", limit=6))


def test_get_recent_messages_returns_only_the_bounded_window_oldest_first():
    store = InMemoryConversationStore()
    conversation = asyncio.run(store.create())

    async def scenario():
        for i in range(5):
            await store.append_message(conversation.id, _msg("user", f"message {i}"))
        return await store.get_recent_messages(conversation.id, limit=2)

    messages = asyncio.run(scenario())

    assert [m.content for m in messages] == ["message 3", "message 4"]


def test_get_recent_messages_returns_everything_when_fewer_than_limit():
    store = InMemoryConversationStore()
    conversation = asyncio.run(store.create())

    async def scenario():
        await store.append_message(conversation.id, _msg("user", "only message"))
        return await store.get_recent_messages(conversation.id, limit=10)

    messages = asyncio.run(scenario())

    assert len(messages) == 1


def test_get_recent_messages_with_zero_limit_returns_empty_list():
    store = InMemoryConversationStore()
    conversation = asyncio.run(store.create())

    async def scenario():
        await store.append_message(conversation.id, _msg("user", "hi"))
        return await store.get_recent_messages(conversation.id, limit=0)

    messages = asyncio.run(scenario())

    assert messages == []


def test_conversations_are_isolated_from_each_other():
    store = InMemoryConversationStore()

    async def scenario():
        conv_a = await store.create()
        conv_b = await store.create()
        await store.append_message(conv_a.id, _msg("user", "hello from A"))
        await store.append_message(conv_b.id, _msg("user", "hello from B"))
        a_messages = await store.get_recent_messages(conv_a.id, limit=10)
        b_messages = await store.get_recent_messages(conv_b.id, limit=10)
        return a_messages, b_messages

    a_messages, b_messages = asyncio.run(scenario())

    assert [m.content for m in a_messages] == ["hello from A"]
    assert [m.content for m in b_messages] == ["hello from B"]
