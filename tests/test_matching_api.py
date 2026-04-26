from __future__ import annotations

import pytest
from httpx import AsyncClient

from tests.conftest import create_user


@pytest.mark.asyncio
async def test_get_recommendations(async_client: AsyncClient):
    user_a = await create_user(async_client, name="Alice")
    user_b = await create_user(async_client, name="Bob")

    resp = await async_client.get(f"/api/users/{user_a['id']}/matches")
    assert resp.status_code == 200
    matches = resp.json()
    assert len(matches) == 1
    assert matches[0]["candidate"]["id"] == user_b["id"]


@pytest.mark.asyncio
async def test_like_user(async_client: AsyncClient):
    user_a = await create_user(async_client, name="Alice")
    user_b = await create_user(async_client, name="Bob")

    resp = await async_client.post(
        f"/api/users/{user_a['id']}/like/{user_b['id']}"
    )
    assert resp.status_code == 201
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_self_like_rejected(async_client: AsyncClient):
    user = await create_user(async_client, name="Alice")
    resp = await async_client.post(
        f"/api/users/{user['id']}/like/{user['id']}"
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_duplicate_like_rejected(async_client: AsyncClient):
    user_a = await create_user(async_client, name="Alice")
    user_b = await create_user(async_client, name="Bob")

    await async_client.post(f"/api/users/{user_a['id']}/like/{user_b['id']}")
    resp = await async_client.post(
        f"/api/users/{user_a['id']}/like/{user_b['id']}"
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_like_nonexistent_user(async_client: AsyncClient):
    user = await create_user(async_client, name="Alice")
    resp = await async_client.post(
        f"/api/users/{user['id']}/like/nonexistent"
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_mutual_match_detected(async_client: AsyncClient):
    user_a = await create_user(async_client, name="Alice")
    user_b = await create_user(async_client, name="Bob")

    await async_client.post(f"/api/users/{user_a['id']}/like/{user_b['id']}")
    await async_client.post(f"/api/users/{user_b['id']}/like/{user_a['id']}")

    matches = await async_client.get(f"/api/users/{user_a['id']}/matches")
    results = matches.json()
    assert results[0]["mutual"] is True
    assert results[0]["liked_by_user"] is True
    assert results[0]["liked_you"] is True


@pytest.mark.asyncio
async def test_mutual_matches_list(async_client: AsyncClient):
    user_a = await create_user(async_client, name="Alice")
    user_b = await create_user(async_client, name="Bob")

    await async_client.post(f"/api/users/{user_a['id']}/like/{user_b['id']}")
    await async_client.post(f"/api/users/{user_b['id']}/like/{user_a['id']}")

    resp = await async_client.get(f"/api/users/{user_a['id']}/mutual")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == user_b["id"]


@pytest.mark.asyncio
async def test_no_mutual_without_reciprocation(async_client: AsyncClient):
    user_a = await create_user(async_client, name="Alice")
    user_b = await create_user(async_client, name="Bob")

    await async_client.post(f"/api/users/{user_a['id']}/like/{user_b['id']}")

    resp = await async_client.get(f"/api/users/{user_a['id']}/mutual")
    assert resp.status_code == 200
    assert len(resp.json()) == 0


@pytest.mark.asyncio
async def test_simulation_endpoint(async_client: AsyncClient):
    resp = await async_client.post("/api/simulation", json={"count": 5})
    assert resp.status_code == 201
    users = resp.json()
    assert len(users) == 5
    for user in users:
        assert user["agent_generated"] is True
