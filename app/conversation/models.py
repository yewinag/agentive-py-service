import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """One turn's worth of content in a Conversation. Deliberately
    minimal: role + content + creation order is everything the bounded
    context strategy and prompt construction need. No user/auth ids, no
    token counts, no tool-call state - those belong to later steps.
    """

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Conversation(BaseModel):
    """An ordered sequence of messages, identified by an opaque id a
    client can pass back on a follow-up request. `id` is generated here
    (a random UUID, not sequential) - a client never invents its own
    conversation_id, and there is no separate internal-vs-public id
    layer because nothing today (no auth, no multi-tenancy) needs one.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[ConversationMessage] = Field(default_factory=list)
