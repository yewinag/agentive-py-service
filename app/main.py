from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.exception_handlers import DEPENDENCY_FAILURE_EXCEPTIONS, handle_dependency_failure
from app.api.router import api_router
from app.core.config import get_settings
from app.knowledge.bootstrap import bootstrap_default_knowledge_base


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Explicit, one-time, isolated from ChatService: populates the
    # default in-memory VectorStore with the project's committed PDFs
    # so /api/v1/chat has something to retrieve against locally. A no-op
    # when VECTOR_STORE_PROVIDER=pgvector - see bootstrap_default_knowledge_base().
    await bootstrap_default_knowledge_base(get_settings())
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
    app.include_router(api_router)
    for exception_type in DEPENDENCY_FAILURE_EXCEPTIONS:
        app.add_exception_handler(exception_type, handle_dependency_failure)
    return app


app = create_app()
