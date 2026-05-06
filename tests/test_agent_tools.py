"""Agent Tool Use 单元 + 集成测试。"""
from __future__ import annotations

import json
import pytest

from app.services.agent_tools import (
    TOOL_SCHEMAS,
    build_tool_dispatch,
    get_my_recommendations,
    safe_parse_arguments,
    search_similar_users,
    update_my_boundary,
)
from tests.conftest import FakeLLM, make_user_with_agent


# ---------- 工具单元测试 ----------

@pytest.mark.asyncio
async def test_tool_schemas_have_required_fields():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"search_similar_users", "get_my_recommendations", "update_my_boundary"}
    for tool in TOOL_SCHEMAS:
        assert tool["type"] == "function"
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]


@pytest.mark.asyncio
async def test_search_similar_users_finds_by_interest(session):
    me, _ = await make_user_with_agent(session, username="me", agent_name="Me", interests=["阅读"])
    await make_user_with_agent(session, username="otaku", agent_name="木木", interests=["动漫", "游戏"])
    await make_user_with_agent(session, username="bookworm", agent_name="阿乐", interests=["阅读", "电影"])
    await session.commit()

    result = await search_similar_users(session=session, current_user_id=me.id, keyword="动漫")
    assert result["match_count"] == 1
    assert result["matches"][0]["name"] == "木木"

    # 排除自己
    result_self = await search_similar_users(session=session, current_user_id=me.id, keyword="阅读")
    names = [m["name"] for m in result_self["matches"]]
    assert "Me" not in names
    assert "阿乐" in names


@pytest.mark.asyncio
async def test_search_similar_users_empty_keyword_lists_community(session):
    """空关键字现在表示'列表模式'：返回任意若干 Agent，方便用户问'有谁'。"""
    me, _ = await make_user_with_agent(session, username="me")
    await make_user_with_agent(session, username="o", agent_name="木木")
    await make_user_with_agent(session, username="b", agent_name="阿乐")
    await session.commit()
    result = await search_similar_users(session=session, current_user_id=me.id, keyword="   ")
    assert result["list_mode"] is True
    assert result["match_count"] >= 2
    names = [m["name"] for m in result["matches"]]
    assert "木木" in names
    assert "阿乐" in names
    # 不应包含自己
    assert all(n != "Aria" for n in names)


@pytest.mark.asyncio
async def test_update_my_boundary_appends_and_dedupes(session):
    me, agent = await make_user_with_agent(session, username="me")
    await session.commit()

    r1 = await update_my_boundary(session=session, current_user_id=me.id, item="不接受异地恋")
    assert r1["ok"] is True
    assert "不接受异地恋" in r1["boundaries"]

    r2 = await update_my_boundary(session=session, current_user_id=me.id, item="不接受异地恋")
    assert r2["ok"] is True
    assert "已存在" in r2["message"]
    assert r2["boundaries"].count("不接受异地恋") == 1


@pytest.mark.asyncio
async def test_update_my_boundary_validates(session):
    me, _ = await make_user_with_agent(session, username="me")
    await session.commit()

    r = await update_my_boundary(session=session, current_user_id=me.id, item="")
    assert r["ok"] is False

    r2 = await update_my_boundary(session=session, current_user_id=me.id, item="x" * 200)
    assert r2["ok"] is False


@pytest.mark.asyncio
async def test_get_my_recommendations_empty(session):
    me, _ = await make_user_with_agent(session, username="me")
    await session.commit()
    result = await get_my_recommendations(session=session, current_user_id=me.id)
    assert result["count"] == 0


# ---------- safe_parse_arguments ----------

def test_safe_parse_arguments_handles_strings_and_dicts():
    assert safe_parse_arguments('{"a":1}') == {"a": 1}
    assert safe_parse_arguments({"a": 1}) == {"a": 1}
    assert safe_parse_arguments("not json") == {}
    assert safe_parse_arguments(None) == {}


# ---------- chat_with_tools 集成测试（FakeLLM） ----------

@pytest.mark.asyncio
async def test_chat_with_tools_executes_search(session):
    me, _ = await make_user_with_agent(session, username="me", agent_name="Me")
    await make_user_with_agent(session, username="otaku", agent_name="木木", interests=["动漫"])
    await session.commit()

    fake = FakeLLM()
    # 第一次返回 tool_calls，要求调用 search_similar_users("动漫")
    fake.enqueue(
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
    # 第二次根据工具结果返回最终自然语言
    fake.enqueue(content="社区里有 木木 喜欢动漫，要不要让我去聊聊？")

    dispatch = build_tool_dispatch(session=session, current_user_id=me.id)
    reply = await fake.chat_with_tools(
        [{"role": "user", "content": "有没有像我这种喜欢动漫的人？"}],
        tools=TOOL_SCHEMAS,
        tool_dispatch=dispatch,
        max_rounds=3,
    )
    assert "木木" in reply
    # 应该至少有 2 次 chat_raw 调用（一次 tool_calls 一次自然语言）
    raw_calls = [c for c in fake.calls if c["kind"] == "chat_raw"]
    assert len(raw_calls) == 2
    # 第二次调用的 messages 中应该有 tool role 消息
    second_messages = raw_calls[1]["messages"]
    tool_role_msgs = [m for m in second_messages if m.get("role") == "tool"]
    assert len(tool_role_msgs) == 1
    parsed = json.loads(tool_role_msgs[0]["content"])
    assert parsed["match_count"] == 1


@pytest.mark.asyncio
async def test_chat_with_tools_handles_no_tool_call(session):
    me, _ = await make_user_with_agent(session, username="me")
    await session.commit()
    fake = FakeLLM()
    fake.enqueue(content="你好，告诉我你最近在忙什么？")
    dispatch = build_tool_dispatch(session=session, current_user_id=me.id)
    reply = await fake.chat_with_tools(
        [{"role": "user", "content": "嗨"}],
        tools=TOOL_SCHEMAS,
        tool_dispatch=dispatch,
        max_rounds=3,
    )
    assert reply == "你好，告诉我你最近在忙什么？"


@pytest.mark.asyncio
async def test_chat_with_tools_handles_unknown_tool(session):
    me, _ = await make_user_with_agent(session, username="me")
    await session.commit()
    fake = FakeLLM()
    fake.enqueue(
        tool_calls=[
            {
                "id": "x",
                "type": "function",
                "function": {"name": "no_such_tool", "arguments": "{}"},
            }
        ]
    )
    fake.enqueue(content="抱歉，刚才尝试的工具不存在，我直接回答你。")
    dispatch = build_tool_dispatch(session=session, current_user_id=me.id)
    reply = await fake.chat_with_tools(
        [{"role": "user", "content": "?"}],
        tools=TOOL_SCHEMAS,
        tool_dispatch=dispatch,
        max_rounds=3,
    )
    assert "抱歉" in reply


@pytest.mark.asyncio
async def test_chat_with_tools_multi_round_boundary_then_search(session):
    me, _ = await make_user_with_agent(session, username="me")
    await make_user_with_agent(session, username="hi", agent_name="小美", interests=["旅行"])
    await session.commit()
    fake = FakeLLM()
    # round 1: update_my_boundary
    fake.enqueue(
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "update_my_boundary",
                    "arguments": json.dumps({"item": "不接受吸烟"}),
                },
            }
        ]
    )
    # round 2: search_similar_users
    fake.enqueue(
        tool_calls=[
            {
                "id": "c2",
                "type": "function",
                "function": {
                    "name": "search_similar_users",
                    "arguments": json.dumps({"keyword": "旅行"}),
                },
            }
        ]
    )
    # round 3: 最终自然语言
    fake.enqueue(content="已记录边界，并找到 小美 喜欢旅行。")

    dispatch = build_tool_dispatch(session=session, current_user_id=me.id)
    reply = await fake.chat_with_tools(
        [{"role": "user", "content": "我不接受吸烟，顺便看看有没有喜欢旅行的人"}],
        tools=TOOL_SCHEMAS,
        tool_dispatch=dispatch,
        max_rounds=4,
    )
    assert "小美" in reply
    # 验证边界已写入 DB
    from sqlalchemy import select
    from app.models.models import Agent

    ag = (await session.execute(select(Agent).where(Agent.user_id == me.id))).scalar_one()
    assert "不接受吸烟" in (ag.personality.get("boundaries") or [])
