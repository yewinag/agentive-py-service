class DocumentExtractionError(Exception):
    """Raised when a DocumentExtractor fails to produce text from a source.
    Extraction-library-specific exceptions are translated into this at the
    extraction boundary, the same way LLMProviderError translates OpenAI
    SDK errors.
    """


class EmbeddingProviderError(Exception):
    """Raised when an EmbeddingProvider fails to produce embeddings.
    Provider SDK/API exceptions are translated into this at the provider
    boundary, the same way LLMProviderError translates OpenAI SDK errors
    for chat.
    """
