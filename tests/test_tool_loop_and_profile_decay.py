"""测试本次修复：工具去重 / 新工具 / 清空聊天 / 画像 LRU 衰减。"""
from __future__ import annotations

import base64
import json
import os

import itsdangerous
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.base import Base
from app.models.models import Agent, Message
from app.services.agent_tools import (
    TOOL_SCHEMAS,
    build_tool_dispatch,
    forget_memory,
    get_community_stats,
    get_my_profile,
)
from tests.conftest import FakeLLM, make_user_with_agent

SESSION_SECRET = "z" * 64


def _build_session_cookie(user_id: int) -> str:
    payload = {"user_id": user_id}
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8"))
    signer = itsdangerous.TimestampSigner(SESSION_SECRET)
    return signer.sign(encoded).decode("utf-8")


@pytest.mark.asyncio
async def test_chat_with_tools_deduplicates_repeat_calls(session):
    me, _ = await make_user_with_agent(session, username="me")
    await make_user_with_agent(session, username="o", agent_name="木木")
    await session.commit()

    fake = FakeLLM()
    tool_call_payload = [
        {
            "id": "c1",
            "type": "function",
            "function": {
                "name": "search_similar_users",
                "arguments": json.dumps({"keyword": ""}),
            },
        }
    ]
    fake.enqueue(tool_calls=tool_call_payload)
    fake.enqueue(tool_calls=tool_call_payload)
    fake.enqueue(tool_calls=tool_call_payload)

    dispatch = build_tool_dispatch(session=session, current_user_id=me.id)
    text, traces = await fake.chat_with_tools_traced(
        [{"role": "user", "content": "有谁"}],
        tools=TOOL_SCHEMAS,
        tool_dispatch=dispatch,
        max_rounds=3,
    )
    assert len(traces) == 1
    assert traces[0]["name"] == "search_similar_users"


@pytest.mark.asyncio
async def test_forget_memory_removes_from_interests(session):
    me, agent = await make_user_with_agent(
        session, username="me", interests=["旅行", "摄影", "动漫"], traits=["内向"]
    )
    await session.commit()
    r = await forget_memory(session=session, current_user_id=me.id, item="动漫")
    assert r["ok"] is True
    assert any(rm["text"] == "动漫" for rm in r["removed"])
    refreshed = (await session.execute(select(Agent).where(Agent.id == agent.id))).scalar_one()
    assert "动漫" not in refreshed.personality["interests"]
    assert "旅行" in refreshed.personality["interests"]


@pytest.mark.asyncio
async def test_forget_memory_not_found(session):
    me, _ = await make_user_with_agent(session, username="me", interests=["旅行"])
    await session.commit()
    r = await forget_memory(session=session, current_user_id=me.id, item="飞行")
    assert r["ok"] is False
    assert "没找到" in r["message"]


@pytest.mark.asyncio
async def test_get_my_profile_strips_internal_fields(session):
    me, agent = await make_user_with_agent(session, username="me", traits=["温柔"], interests=["阅读"])
    profile = dict(agent.personality)
    profile["_embedding_vector"] = [0.1, 0.2]
    profile["_secret"] = "should not leak"
    agent.personality = profile
    await session.commit()

    r = await get_my_profile(session=session, current_user_id=me.id)
    assert r["ok"] is True
    assert r["traits"] == ["温柔"]
    assert r["interests"] == ["阅读"]
    for key in r:
        assert not str(key).startswith("_")


@pytest.mark.asyncio
async def test_get_community_stats(session):
    me, _ = await make_user_with_agent(session, username="me", agent_name="A", vibe="温暖治愈")
    await make_user_with_agent(session, username="b", agent_name="B", vibe="温暖治愈")
    await make_user_with_agent(session, username="c", agent_name="C", vibe="阳光活力")
    await session.commit()

    r = await get_community_stats(session=session, current_user_id=me.id)
    assert r["ok"] is True
    assert r["total_agents"] == 3
    top = {item["vibe"]: item["count"] for item in r["top_vibes"]}
    assert top.get("温暖治愈") == 2
    assert top.get("阳光活力") == 1


def test_profile_lru_pushes_old_items_out():
    from app.main import _merge_profile

    # 用真实风格的不同兴趣词，避免被 bigram 语义去重合并，单纯验证 LRU + cap
    old_interests = ["运动", "读书", "音乐", "电影", "旅行", "美食", "摄影", "游戏", "动漫", "编程"]
    old = {"interests": old_interests, "traits": []}
    new = {"interests": ["写作"]}
    merged = _merge_profile(old, new)
    assert merged["interests"][0] == "写作"
    assert len(merged["interests"]) == 10
    assert "编程" not in merged["interests"]  # 最旧的被挤出


def test_profile_lru_keeps_new_first():
    from app.main import _merge_profile

    old = {"interests": ["旅行", "摄影"]}
    new = {"interests": ["阅读"]}
    merged = _merge_profile(old, new)
    assert merged["interests"][0] == "阅读"
    assert "旅行" in merged["interests"]


@pytest_asyncio.fixture
async def app_client():
    os.environ["DEBUG"] = "true"
    os.environ["SESSION_SECRET"] = SESSION_SECRET
    os.environ["EMAIL_CODE_SECRET"] = "y" * 64
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"
    from app.main import create_app

    app = create_app()
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    app.state.engine = engine
    app.state.session_factory = factory
    fake_llm = FakeLLM()
    app.state.llm = fake_llm

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, app, fake_llm
    await engine.dispose()


@pytest.mark.asyncio
async def test_chat_clear_endpoint_removes_messages(app_client):
    client, app, fake_llm = app_client
    factory = app.state.session_factory
    async with factory() as s:
        me, my_agent = await make_user_with_agent(s, username="alice", agent_name="A")
        await s.commit()
        agent_id = my_agent.id
        user_id = me.id

    fake_llm.enqueue(content="嗨")
    fake_llm.enqueue(content="再来一句")
    cookie = _build_session_cookie(user_id)
    for msg in ["你好", "今天天气怎样"]:
        await client.post(
            f"/chat/{agent_id}",
            data={"content": msg},
            headers={"X-Requested-With": "XMLHttpRequest", "Origin": "http://testserver"},
            cookies={"session": cookie},
        )

    async with factory() as s:
        rows = (await s.execute(select(Message))).scalars().all()
        assert len(rows) >= 4

    resp = await client.post(
        f"/chat/{agent_id}/clear",
        headers={"X-Requested-With": "XMLHttpRequest", "Origin": "http://testserver"},
        cookies={"session": cookie},
    )
    assert resp.status_code == 200, resp.text[:300]
    body = resp.json()
    assert body["ok"] is True
    assert body["deleted_count"] >= 4

    async with factory() as s:
        remaining = (await s.execute(select(Message))).scalars().all()
        assert len(remaining) == 0


@pytest.mark.asyncio
async def test_chat_clear_unauthenticated_rejected(app_client):
    client, app, _ = app_client
    factory = app.state.session_factory
    async with factory() as s:
        _, my_agent = await make_user_with_agent(s, username="alice", agent_name="A")
        await s.commit()
        agent_id = my_agent.id

    resp = await client.post(
        f"/chat/{agent_id}/clear",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_chat_clear_other_user_agent_forbidden(app_client):
    client, app, _ = app_client
    factory = app.state.session_factory
    async with factory() as s:
        me, _ = await make_user_with_agent(s, username="alice", agent_name="A")
        _, other_agent = await make_user_with_agent(s, username="bob", agent_name="B")
        await s.commit()
        other_agent_id = other_agent.id
        user_id = me.id

    cookie = _build_session_cookie(user_id)
    resp = await client.post(
        f"/chat/{other_agent_id}/clear",
        headers={"X-Requested-With": "XMLHttpRequest", "Origin": "http://testserver"},
        cookies={"session": cookie},
    )
    assert resp.status_code == 404
