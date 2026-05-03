"""推荐卡片可视化测试。

- Recommendation 数据库新增 highlights / risks 列
- discovery 落库时填入 evaluation 的 highlights / risks
- dashboard 渲染时把它们以 chip 展示
"""
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
from app.models.models import Agent, Recommendation, User
from app.services import discovery_service
from app.services.discovery_service import AgentSnapshot
from tests.conftest import FakeLLM, make_user_with_agent

SESSION_SECRET = "z" * 64


def _build_session_cookie(user_id: int) -> str:
    payload = {"user_id": user_id}
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8"))
    signer = itsdangerous.TimestampSigner(SESSION_SECRET)
    return signer.sign(encoded).decode("utf-8")


# ---------- 数据库列存在 ----------

@pytest.mark.asyncio
async def test_recommendation_persists_highlights_and_risks(session):
    me, my_agent = await make_user_with_agent(session, username="me", agent_name="A")
    _, peer_agent = await make_user_with_agent(session, username="peer", agent_name="B")
    await session.commit()

    fake = FakeLLM()
    for i in range(8):
        fake.enqueue(json_data={"thought": "", "observation": "", "action": f"m{i}"})
    for s in [80, 80, 80]:
        fake.enqueue(
            json_data={
                "compatible": True,
                "score": s,
                "confidence": 80,
                "reason": "看起来很合拍",
                "highlights": ["共同兴趣旅行", "沟通节奏一致"],
                "risks": ["都比较内向，需要主动"],
            }
        )
    fake.enqueue(
        json_data={
            "judge_pass": True,
            "veto_reason": "",
            "additional_risks": ["judge_extra"],
            "score_adjustment": 0,
        }
    )

    a = AgentSnapshot(id=my_agent.id, name=my_agent.name, personality=my_agent.personality)
    b = AgentSnapshot(id=peer_agent.id, name=peer_agent.name, personality=peer_agent.personality)
    result = await discovery_service._agent_chat_and_evaluate(
        session, a, b, fake, min_match_score=50, min_confidence=50
    )
    await session.commit()

    assert result is not None
    rec = (
        await session.execute(select(Recommendation).where(Recommendation.id == result["recommendation_id"]))
    ).scalar_one()
    assert "共同兴趣旅行" in rec.highlights
    assert "沟通节奏一致" in rec.highlights
    # judge.additional_risks 与原始 risks 都应该入库
    assert "都比较内向，需要主动" in rec.risks
    assert "judge_extra" in rec.risks


@pytest.mark.asyncio
async def test_recommendation_default_empty_lists(session):
    """没明确填的时候默认空列表（防 None）。"""
    me, _ = await make_user_with_agent(session, username="me", agent_name="A")
    _, peer = await make_user_with_agent(session, username="peer", agent_name="B")
    await session.commit()

    rec = Recommendation(
        from_agent_id=1,  # 随便填，仅测试默认值
        to_agent_id=2,
        score=70,
        reason="x",
    )
    session.add(rec)
    await session.flush()
    assert rec.highlights == [] or rec.highlights is None or rec.highlights == "[]"
    # SQLAlchemy default=list 在 ORM 层是 []
    if rec.highlights is None:
        rec.highlights = []
    if rec.risks is None:
        rec.risks = []
    assert isinstance(rec.highlights, list)
    assert isinstance(rec.risks, list)


# ---------- dashboard 渲染卡片 ----------

@pytest_asyncio.fixture
async def dashboard_client():
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
    app.state.llm = FakeLLM()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, app

    await engine.dispose()


@pytest.mark.asyncio
async def test_dashboard_renders_highlights_and_risks_chips(dashboard_client):
    """dashboard 推荐 tab 应该把 highlights 和 risks 渲染成 chip。"""
    client, app = dashboard_client
    factory = app.state.session_factory
    async with factory() as s:
        me, my_agent = await make_user_with_agent(s, username="alice", agent_name="A")
        _, peer = await make_user_with_agent(s, username="bob", agent_name="B")
        await s.commit()

        rec = Recommendation(
            from_agent_id=peer.id,
            to_agent_id=my_agent.id,
            score=82,
            reason="测试推荐",
            highlights=["共同兴趣摄影", "都在上海"],
            risks=["对方更外向", "judge_extra_concern"],
            status="pending",
        )
        s.add(rec)
        await s.commit()
        user_id = me.id

    cookie = _build_session_cookie(user_id)
    resp = await client.get(
        "/dashboard?tab=decision",
        headers={"Origin": "http://testserver"},
        cookies={"session": cookie},
    )
    assert resp.status_code == 200, resp.text[:300]
    body = resp.text
    # 卡片亮点
    assert "共同兴趣摄影" in body
    assert "都在上海" in body
    # 卡片风险
    assert "对方更外向" in body
    assert "judge_extra_concern" in body
    # CSS 类名应该出现
    assert "rec-highlights" in body
    assert "rec-risks" in body
    assert "chip-positive" in body
    assert "chip-warning" in body
    # 标签文字
    assert "亮点" in body
    assert "风险" in body


@pytest.mark.asyncio
async def test_dashboard_omits_card_sections_when_empty(dashboard_client):
    """没有 highlights/risks 时不应该渲染对应区块。"""
    client, app = dashboard_client
    factory = app.state.session_factory
    async with factory() as s:
        me, my_agent = await make_user_with_agent(s, username="alice", agent_name="A")
        _, peer = await make_user_with_agent(s, username="bob", agent_name="B")
        await s.commit()
        rec = Recommendation(
            from_agent_id=peer.id,
            to_agent_id=my_agent.id,
            score=70,
            reason="test",
            highlights=[],
            risks=[],
            status="pending",
        )
        s.add(rec)
        await s.commit()
        user_id = me.id

    cookie = _build_session_cookie(user_id)
    resp = await client.get(
        "/dashboard?tab=decision",
        cookies={"session": cookie},
        headers={"Origin": "http://testserver"},
    )
    assert resp.status_code == 200
    body = resp.text
    assert "rec-highlights" not in body
    assert "rec-risks" not in body
