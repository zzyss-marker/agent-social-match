from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    resp = await async_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_readiness_check(async_client: AsyncClient):
    resp = await async_client.get("/api/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["database"] == "connected"
