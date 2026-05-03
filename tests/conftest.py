"""测试基础设施。

旧测试文件针对的是已经废弃的 API 契约（/api/users 等），与当前代码不兼容；
本 conftest 提供干净的内存 SQLite + Agent 工厂 + FakeLLM mock，
用于覆盖 Tool Use / ReAct / 向量召回 / JudgeAgent / 解释卡片 等新功能。
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.base import Base
from app.models.models import Agent, User


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s


async def make_user_with_agent(
    session: AsyncSession,
    *,
    username: str = "alice",
    agent_name: str = "Aria",
    traits: list[str] | None = None,
    interests: list[str] | None = None,
    looking_for: str = "",
    vibe: str = "自然友好",
) -> tuple[User, Agent]:
    user = User(username=username, email=f"{username}@example.com", email_verified=True)
    session.add(user)
    await session.flush()
    agent = Agent(
        user_id=user.id,
        name=agent_name,
        personality={
            "traits": traits or [],
            "interests": interests or [],
            "looking_for": looking_for,
            "vibe": vibe,
            "context_memory": [],
            "boundaries": [],
            "conversation_style": "",
            "snapshots": [],
        },
        status="idle",
    )
    session.add(agent)
    await session.flush()
    return user, agent


class FakeLLM:
    """可编排的模拟 LLM，按 enqueue 顺序返回响应。

    支持：
    - chat_raw(messages, tools=...) -> dict {role,content,tool_calls?}
    - chat(messages, ...) -> str
    - chat_json(messages, ...) -> dict
    复用 LLMClient.chat_with_tools 时只需要 chat_raw + chat。
    """

    def __init__(self) -> None:
        self.responses: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    def enqueue(
        self,
        *,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> None:
        self.responses.append(
            {"content": content, "tool_calls": tool_calls, "json_data": json_data}
        )

    def _next(self) -> dict[str, Any]:
        if not self.responses:
            return {"content": "", "tool_calls": None, "json_data": None}
        return self.responses.pop(0)

    async def chat_raw(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"kind": "chat_raw", "messages": messages, "kwargs": kwargs})
        resp = self._next()
        out: dict[str, Any] = {"role": "assistant", "content": resp.get("content") or None}
        if resp.get("tool_calls"):
            out["tool_calls"] = resp["tool_calls"]
        return out

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.calls.append({"kind": "chat", "messages": messages, "kwargs": kwargs})
        return self._next().get("content") or ""

    async def chat_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"kind": "chat_json", "messages": messages, "kwargs": kwargs})
        return self._next().get("json_data") or {}

    # Bind chat_with_tools from real LLMClient so we test the actual loop logic
    async def chat_with_tools(self, messages, tools, tool_dispatch, **kwargs):
        from app.services.llm_client import LLMClient

        return await LLMClient.chat_with_tools(self, messages, tools, tool_dispatch, **kwargs)

    # Bind ReAct / Self-Consistency / agent_chat_turn helpers from real LLMClient
    async def agent_chat_turn(self, agent_name, agent_personality, conversation_history):
        from app.services.llm_client import LLMClient

        return await LLMClient.agent_chat_turn(self, agent_name, agent_personality, conversation_history)

    async def agent_chat_turn_react(self, agent_name, agent_personality, conversation_history):
        from app.services.llm_client import LLMClient

        return await LLMClient.agent_chat_turn_react(
            self, agent_name, agent_personality, conversation_history
        )

    async def evaluate_match(self, *args, **kwargs):
        from app.services.llm_client import LLMClient

        return await LLMClient.evaluate_match(self, *args, **kwargs)

    async def evaluate_match_self_consistent(self, *args, **kwargs):
        from app.services.llm_client import LLMClient

        return await LLMClient.evaluate_match_self_consistent(self, *args, **kwargs)

    async def judge_recommendation(self, *args, **kwargs):
        from app.services.llm_client import LLMClient

        return await LLMClient.judge_recommendation(self, *args, **kwargs)
