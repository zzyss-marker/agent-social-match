from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.main import create_app
from app.models.base import Base
from app.schemas.user import UserCreate

TEST_DATABASE_URL = "sqlite+aiosqlite://"


@pytest_asyncio.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    app = create_app()
    app.state.settings.DATABASE_URL = TEST_DATABASE_URL

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    app.state.engine = engine
    app.state.session_factory = session_factory

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_client) -> AsyncGenerator[AsyncSession, None]:
    factory = async_client._transport.app.state.session_factory
    async with factory() as session:
        yield session


def make_test_user(name: str = "testuser", **kwargs) -> UserCreate:
    defaults = {
        "name": name,
        "age": 28,
        "city": "Shanghai",
        "intent": "companion",
        "hobbies": ["hiking", "reading"],
        "values": ["honesty", "growth mindset"],
        "availability": ["weeknight", "weekend morning"],
        "communication_style": "direct",
        "preferred_age_min": 24,
        "preferred_age_max": 35,
        "preferred_city": "Any",
        "accept_remote": True,
        "description": "Test user",
    }
    defaults.update(kwargs)
    return UserCreate(**defaults)


async def create_user(client: AsyncClient, **kwargs) -> dict:
    user_data = make_test_user(**kwargs)
    resp = await client.post("/api/users", json=user_data.model_dump())
    return resp.json()
