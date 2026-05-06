"""验证 chat_send 的 JSON 响应里包含 tool_traces，前端能据此渲染 chip。"""
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

SESSION_SECRET = "z" * 64


def _build_session_cookie(user_id: int) -> str:
    payload = {"user_id": user_id}
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8"))
    signer = itsdangerous.TimestampSigner(SESSION_SECRET)
    return signer.sign(encoded).decode("utf-8")


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
async def test_chat_response_includes_tool_traces_when_tool_called(app_client):
    """触发 search_similar_users 后，JSON 响应里要带 tool_traces 数组。"""
    client, app, fake_llm = app_client
    factory = app.state.session_factory
    async with factory() as s:
        me, my_agent = await make_user_with_agent(s, username="alice", agent_name="Aria")
        await make_user_with_agent(s, username="o", agent_name="木木", interests=["动漫"])
        await s.commit()
        agent_id = my_agent.id
        user_id = me.id

    fake_llm.enqueue(
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "search_similar_users",
                    "arguments": json.dumps({"keyword": "动漫"}),
                },
            }
        ]
    )
    fake_llm.enqueue(content="木木 喜欢动漫，要不要让我去聊聊？")

    cookie = _build_session_cookie(user_id)
    resp = await client.post(
        f"/chat/{agent_id}",
        data={"content": "有没有像我这种喜欢动漫的人？"},
        headers={"X-Requested-With": "XMLHttpRequest", "Origin": "http://testserver"},
        cookies={"session": cookie},
    )
    assert resp.status_code == 200, resp.text[:300]
    body = resp.json()
    assert "tool_traces" in body
    assert len(body["tool_traces"]) == 1
    trace = body["tool_traces"][0]
    assert trace["name"] == "search_similar_users"
    assert trace["ok"] is True
    assert trace["arguments"] == {"keyword": "动漫"}
    # summary 应该包含 '木木'
    assert "木木" in trace["summary"]


@pytest.mark.asyncio
async def test_chat_response_empty_traces_when_no_tool_call(app_client):
    """未触发工具时 tool_traces 为空数组。"""
    client, app, fake_llm = app_client
    factory = app.state.session_factory
    async with factory() as s:
        me, my_agent = await make_user_with_agent(s, username="alice", agent_name="Aria")
        await s.commit()
        agent_id = my_agent.id
        user_id = me.id

    fake_llm.enqueue(content="嗨，最近怎么样？")
    cookie = _build_session_cookie(user_id)
    resp = await client.post(
        f"/chat/{agent_id}",
        data={"content": "你好"},
        headers={"X-Requested-With": "XMLHttpRequest", "Origin": "http://testserver"},
        cookies={"session": cookie},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tool_traces"] == []


@pytest.mark.asyncio
async def test_chat_response_traces_for_list_mode_search(app_client):
    """问 '系统里有哪些 agent' 触发空关键字搜索（list_mode），summary 应说 '列出社区 Agent'。"""
    client, app, fake_llm = app_client
    factory = app.state.session_factory
    async with factory() as s:
        me, my_agent = await make_user_with_agent(s, username="alice", agent_name="Aria")
        await make_user_with_agent(s, username="o", agent_name="木木", interests=["动漫"])
        await make_user_with_agent(s, username="b", agent_name="阿乐", interests=["阅读"])
        await s.commit()
        agent_id = my_agent.id
        user_id = me.id

    fake_llm.enqueue(
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "search_similar_users",
                    "arguments": json.dumps({"keyword": ""}),
                },
            }
        ]
    )
    fake_llm.enqueue(content="社区里有 木木、阿乐 两位 Agent。")

    cookie = _build_session_cookie(user_id)
    resp = await client.post(
        f"/chat/{agent_id}",
        data={"content": "系统里都有哪些 agent"},
        headers={"X-Requested-With": "XMLHttpRequest", "Origin": "http://testserver"},
        cookies={"session": cookie},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["tool_traces"]) == 1
    trace = body["tool_traces"][0]
    assert trace["arguments"] == {"keyword": ""}
    assert "列出社区" in trace["summary"]
    assert "2" in trace["summary"]  # 找到 2 个
