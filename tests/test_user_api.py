from __future__ import annotations

import pytest
from httpx import AsyncClient

from tests.conftest import create_user, make_test_user


@pytest.mark.asyncio
async def test_create_user(async_client: AsyncClient):
    data = make_test_user(name="Alice")
    resp = await async_client.post("/api/users", json=data.model_dump())
    assert resp.status_code == 201
    result = resp.json()
    assert result["name"] == "Alice"
    assert result["agent_generated"] is False
    assert len(result["id"]) == 10


@pytest.mark.asyncio
async def test_create_user_validation_error(async_client: AsyncClient):
    data = make_test_user(name="Alice")
    data_dict = data.model_dump()
    data_dict["preferred_age_min"] = 40
    data_dict["preferred_age_max"] = 20
    resp = await async_client.post("/api/users", json=data_dict)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_user(async_client: AsyncClient):
    created = await create_user(async_client, name="Bob")
    resp = await async_client.get(f"/api/users/{created['id']}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Bob"


@pytest.mark.asyncio
async def test_get_nonexistent_user(async_client: AsyncClient):
    resp = await async_client.get("/api/users/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_users_pagination(async_client: AsyncClient):
    for i in range(5):
        await create_user(async_client, name=f"User{i}")

    resp = await async_client.get("/api/users?offset=0&limit=3")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 3
    assert data["total"] == 5
    assert data["offset"] == 0
    assert data["limit"] == 3


@pytest.mark.asyncio
async def test_list_users_empty(async_client: AsyncClient):
    resp = await async_client.get("/api/users")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert len(data["items"]) == 0
