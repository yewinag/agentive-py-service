class ConversationNotFoundError(Exception):
    """Raised when an operation needs an existing conversation (appending
    a message, reading its history) but the given conversation_id isn't
    in the store. Distinct from ConversationStore.get(), which returns
    None for a missing id instead of raising - get() is a check, this is
    a precondition failure for an operation that assumes existence.

    ChatService never lets this reach the HTTP layer (it always creates
    or checks-and-falls-back before appending/reading), so it is not
    mapped to a response status - a leak would indicate a real bug, best
    left to the generic 500 rather than disguised as a "known" failure.
    """
