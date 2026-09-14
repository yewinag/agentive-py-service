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


class VectorStoreError(Exception):
    """Raised when a VectorStore fails to store or search records.
    Database/client-specific exceptions are translated into this at the
    storage boundary, the same way EmbeddingProviderError translates
    OpenAI SDK errors.
    """
