"""Phase 2.7.1: proves the three new LangChain dependencies
(langchain, langchain-openai, langchain-qdrant) are installed correctly
and importable in this project's environment (Python 3.9, alongside the
existing openai/qdrant-client/pydantic v2 stack). No LangChain component
is built or used here yet - see app/langchain_integration/__init__.py
for the boundary this package establishes.
"""
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore as LangChainQdrantVectorStore


def test_langchain_core_document_is_importable_and_constructible():
    document = Document(page_content="hello", metadata={"source": "test"})

    assert document.page_content == "hello"
    assert document.metadata == {"source": "test"}


def test_langchain_openai_chat_model_is_importable_and_is_a_base_chat_model():
    assert issubclass(ChatOpenAI, BaseChatModel)


def test_langchain_openai_embeddings_is_importable_and_is_an_embeddings_impl():
    assert issubclass(OpenAIEmbeddings, Embeddings)


def test_langchain_qdrant_vector_store_is_importable():
    # Only proves the class loads cleanly against this project's
    # installed qdrant-client version - it is not instantiated or used
    # anywhere yet. Deliberately imported under an alias distinct from
    # this project's own app.knowledge.qdrant_vector_store.QdrantVectorStore
    # - same concept, two different classes, never to be confused.
    assert LangChainQdrantVectorStore.__name__ == "QdrantVectorStore"
    assert LangChainQdrantVectorStore.__module__.startswith("langchain_qdrant")
