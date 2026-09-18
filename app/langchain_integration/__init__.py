"""Integration boundary for LangChain.

This package is where LangChain-specific code lives: the
`langchain`/`langchain-openai`/`langchain-qdrant` SDKs are
implementation details of whatever lands here, the same way `openai` is
isolated to `app/llm/openai_provider.py` and
`app/knowledge/openai_embedding_provider.py`, and `qdrant-client` is
isolated to `app/knowledge/qdrant_vector_store.py`. Nothing outside this
package should import `langchain*` directly.

Phase 2.7.1 established the package (dependencies only, no code).
Phase 2.7.3 (`retriever.py`) added the first real component:
`KnowledgeBaseRetriever`, a `langchain_core.retrievers.BaseRetriever`
that adapts the existing `Retriever`/`VectorRetriever` to LangChain's
`Document` contract - see that module's docstring for why it wraps the
existing retriever rather than using `langchain_qdrant.QdrantVectorStore`
directly against the existing collection.

Named `langchain_integration`, not `langchain`, deliberately: a
same-named local package would sit in front of the real `langchain`
package on the import path for anything inside `app/`, which is exactly
the kind of accidental shadowing this project's existing provider
boundaries are designed to avoid.

What stays exactly as it is, untouched by this phase or by whatever
LangChain component comes next:
- `app.llm.provider.LLMProvider` / `app.llm.openai_provider.OpenAIProvider`
- `app.knowledge.embedding.EmbeddingProvider` / `OpenAIEmbeddingProvider`
- `app.knowledge.vector_store.VectorStore` / `QdrantVectorStore` / `PgVectorStore`
- `app.knowledge.retriever.VectorRetriever`
- `app.knowledge.ingestion.IngestionService` and the explicit ingestion
  command (`app/knowledge/ingest.py`)
- `app.rag.answer_generator.AnswerGenerator` and `app.agent.service.AgentService`

Qdrant remains the vector database and Strapi/PostgreSQL remains the
business source of truth regardless of what orchestration layer sits
above them - LangChain is another way to call these same systems, not a
replacement for what owns their data.
"""
