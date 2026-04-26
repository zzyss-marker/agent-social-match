from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.database import get_async_session


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_llm(request: Request):
    return request.app.state.llm


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    factory = request.app.state.session_factory
    async for session in get_async_session(factory):
        yield session
