from fastapi import APIRouter

from app.chats.router import router as chat_router
from app.health.router import router as health_router

# Unversioned, infra-facing routes (health checks, readiness probes).
api_router = APIRouter()
api_router.include_router(health_router)

# Versioned business/feature routes. Every future feature router
# (rag, retrieval, ...) registers here, so the /api/v1 prefix stays
# in one place instead of being repeated per module.
v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(chat_router)

api_router.include_router(v1_router)
