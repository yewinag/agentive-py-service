class LLMProviderError(Exception):
    """Raised when an LLMProvider fails to produce a reply. Provider SDK
    exceptions are translated into this at the provider boundary, so
    ChatService and the router never see a provider-specific exception type.
    """
