from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.health.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", environment=settings.environment)
