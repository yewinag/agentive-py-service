"""LangChain Retriever adapter over this project's EXISTING knowledge
retrieval pipeline (VectorRetriever -> EmbeddingProvider ->
QdrantVectorStore -> the real "knowledge_chunk_embeddings" collection).

Introduces LangChain's Retriever abstraction (`BaseRetriever`) as a
translation boundary only: the actual embedding call, Qdrant query, and
similarity ranking are entirely delegated to the existing, unmodified
`Retriever`/`VectorRetriever` (see app/knowledge/retriever.py). This
module adds no retrieval logic of its own - it exists purely so a
LangChain-shaped caller can get back real `langchain_core.documents.
Document` objects, sourced from the same collection, embedding model,
and Qdrant client production retrieval already uses.

Why not langchain_qdrant.QdrantVectorStore directly (inspected before
writing this, per this phase's own instruction to check first):
QdrantVectorStore._document_from_point() (langchain_qdrant/qdrant.py)
builds Document.metadata from exactly one payload key
(`metadata_payload_key`, default "metadata"), expecting that key's
*value* to already be a nested dict - e.g. {"page_content": "...",
"metadata": {...}}. This project's existing, already-populated
collection stores a flat payload instead (see
app/knowledge/qdrant_vector_store.py:_chunk_to_payload):
`chunk_id`/`document_id`/`document_title`/`section_heading`/`text`/
`position` all sit directly on the payload, not nested under a
"metadata" key. Pointing langchain_qdrant.QdrantVectorStore's search
methods at this collection unmodified would silently return an empty
`metadata` dict for every result - the only fixes would be re-ingesting
under a different payload shape or monkeypatching a private helper
method, and both are out of scope this phase (no re-ingestion, no
QdrantVectorStore rewrite). Wrapping the existing Retriever instead
needs neither: it already returns every field correctly, and this
module's only job is the Document translation LangChain expects.
"""
from typing import Any, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import Settings
from app.knowledge.vector_store import VectorSearchResult


class KnowledgeBaseRetriever(BaseRetriever):
    """Adapts this project's existing Retriever to LangChain's
    BaseRetriever contract. `retriever` is typed `Any` rather than the
    `Retriever` Protocol it actually satisfies: `Retriever` is a plain
    (non-`@runtime_checkable`) Protocol, and BaseRetriever is a pydantic
    model - validating a non-runtime-checkable Protocol type at
    construction time would raise, not because the wrong kind of object
    was passed.
    """

    retriever: Any
    top_k: Optional[int] = None

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = await self.retriever.retrieve(query, top_k=self.top_k)
        return [_to_document(result) for result in results]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # This project's Retriever Protocol is async-only (see
        # app/knowledge/retriever.py) - every real caller (FastAPI
        # request handlers, AgentService, this integration's own tests)
        # is already async. Raising here is more honest than silently
        # blocking the event loop with asyncio.run() from inside a sync
        # call, or double-running a loop if one is already active.
        raise NotImplementedError(
            "KnowledgeBaseRetriever is async-only - call ainvoke()/aget_relevant_documents(), "
            "matching this project's existing async-only Retriever Protocol."
        )


def _to_document(result: VectorSearchResult) -> Document:
    """The LangChain Document translation boundary: every existing
    DocumentChunk field is preserved in `metadata`, plus the similarity
    `score` VectorSearchResult carries alongside it - nothing is
    dropped, nothing is renamed to fit a LangChain-specific convention.
    """
    chunk = result.chunk
    return Document(
        page_content=chunk.text,
        metadata={
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "document_title": chunk.document_title,
            "section_heading": chunk.section_heading,
            "position": chunk.position,
            "score": result.score,
        },
    )


def get_langchain_retriever(settings: Settings, top_k: Optional[int] = None) -> KnowledgeBaseRetriever:
    """Composition point, matching every other app/knowledge composition
    function's plain-function pattern (get_retriever, get_vector_store,
    get_embedding_provider). Builds the existing VectorRetriever exactly
    as get_retriever(settings) already does - the real EmbeddingProvider
    and QdrantVectorStore Settings selects - so this always queries the
    SAME collection and embedding model production retrieval uses, never
    a second, parallel configuration.
    """
    from app.knowledge.retriever import get_retriever

    return KnowledgeBaseRetriever(retriever=get_retriever(settings), top_k=top_k)
