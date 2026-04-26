from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_settings
from app.core.config import Settings
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        database="unknown",
    )


@router.get("/health/ready", response_model=HealthResponse)
async def readiness_check(
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_db),
) -> HealthResponse:
    try:
        await session.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    ready = db_status == "connected"
    return HealthResponse(
        status="healthy" if ready else "degraded",
        version=settings.APP_VERSION,
        database=db_status,
    )
