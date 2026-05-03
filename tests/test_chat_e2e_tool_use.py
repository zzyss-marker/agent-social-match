"""端到端 HTTP 测试：模拟用户在浏览器点击发送按钮。

用 itsdangerous 直接生成 SessionMiddleware 兼容的签名 cookie，免去登录链路依赖。
"""
from __future__ import annotations

import base64
import json
import os

import itsdangerous
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.base import Base
from tests.conftest import FakeLLM, make_user_with_agent

SESSION_SECRET = "x" * 64


def _build_session_cookie(user_id: int) -> str:
    """生成 SessionMiddleware 兼容的签名 session cookie。

    starlette SessionMiddleware 默认用 itsdangerous.TimestampSigner，
    cookie 值 = signer.sign(base64(json(payload)))。
    """
    payload = {"user_id": user_id}
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8"))
    signer = itsdangerous.TimestampSigner(SESSION_SECRET)
    signed = signer.sign(encoded)
    return signed.decode("utf-8")


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
async def test_unauthenticated_chat_post_returns_401_json(app_client):
    """点击发送按钮时，未登录返回 401（带 X-Requested-With）。"""
    client, app, _ = app_client
    factory = app.state.session_factory
    async with factory() as s:
        _, my_agent = await make_user_with_agent(s, username="alice", agent_name="Aria")
        await s.commit()
        agent_id = my_agent.id

    resp = await client.post(
        f"/chat/{agent_id}",
        data={"content": "hi"},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert resp.status_code == 401
    assert resp.json()["ok"] is False


@pytest.mark.asyncio
async def test_logged_in_chat_button_triggers_tool_call_e2e(app_client):
    """已登录用户点击发送按钮：完整链路 POST /chat/{agent_id} → 工具调用 → 返回带工具结果的回复。"""
    client, app, fake_llm = app_client
    factory = app.state.session_factory
    async with factory() as s:
        me, my_agent = await make_user_with_agent(s, username="alice", agent_name="Aria")
        await make_user_with_agent(s, username="otaku", agent_name="木木", interests=["动漫"])
        await s.commit()
        agent_id = my_agent.id
        user_id = me.id

    # LLM 流水：1) tool_call → 2) 自然语言总结
    fake_llm.enqueue(
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search_similar_users",
                    "arguments": json.dumps({"keyword": "动漫"}),
                },
            }
        ]
    )
    fake_llm.enqueue(content="社区里有 木木 喜欢动漫，要不要让我去聊聊？")

    cookie = _build_session_cookie(user_id)
    resp = await client.post(
        f"/chat/{agent_id}",
        data={"content": "有没有像我这种喜欢动漫的人？"},
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "http://testserver",
        },
        cookies={"session": cookie},
    )

    assert resp.status_code == 200, f"期望 200，实际 {resp.status_code}: {resp.text[:300]}"
    body = resp.json()
    assert body["ok"] is True
    assert "木木" in body["agent_message"]["content"]


@pytest.mark.asyncio
async def test_logged_in_chat_no_tool_call_path(app_client):
    """没有触发工具的普通聊天也要正常工作。"""
    client, app, fake_llm = app_client
    factory = app.state.session_factory
    async with factory() as s:
        me, my_agent = await make_user_with_agent(s, username="bob", agent_name="Bob")
        await s.commit()
        agent_id = my_agent.id
        user_id = me.id

    fake_llm.enqueue(content="嗨，最近在忙什么？")

    cookie = _build_session_cookie(user_id)
    resp = await client.post(
        f"/chat/{agent_id}",
        data={"content": "你好"},
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "http://testserver",
        },
        cookies={"session": cookie},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["agent_message"]["content"] == "嗨，最近在忙什么？"
