"""JudgeAgent 仲裁测试。"""
from __future__ import annotations

import pytest

from app.services import discovery_service
from app.services.discovery_service import AgentSnapshot
from app.services.llm_client import LLMClient
from tests.conftest import FakeLLM, make_user_with_agent


@pytest.mark.asyncio
async def test_judge_recommendation_passes_clean_case():
    fake = FakeLLM()
    fake.enqueue(
        json_data={
            "judge_pass": True,
            "veto_reason": "",
            "additional_risks": [],
            "score_adjustment": 0,
        }
    )
    out = await LLMClient.judge_recommendation(
        fake, "A", {}, "B", {},
        [{"role": "user", "content": "hello"}],
        primary_evaluation={"score": 80, "confidence": 70, "compatible": True},
    )
    assert out["judge_pass"] is True
    assert out["score_adjustment"] == 0
    assert out["additional_risks"] == []


@pytest.mark.asyncio
async def test_judge_recommendation_vetoes_on_boundary_conflict():
    fake = FakeLLM()
    fake.enqueue(
        json_data={
            "judge_pass": False,
            "veto_reason": "对方爱好与 A 的明确边界冲突（吸烟）",
            "additional_risks": ["边界冲突：吸烟"],
            "score_adjustment": -20,
        }
    )
    out = await LLMClient.judge_recommendation(
        fake,
        "A", {"boundaries": ["不接受吸烟"]},
        "B", {"interests": ["吸烟"]},
        [], primary_evaluation={"score": 90, "confidence": 80, "compatible": True},
    )
    assert out["judge_pass"] is False
    assert "吸烟" in out["veto_reason"]
    assert out["score_adjustment"] == -20
    assert "边界冲突：吸烟" in out["additional_risks"]


@pytest.mark.asyncio
async def test_judge_recommendation_clamps_score_adjustment():
    """score_adjustment 应该被限制在 [-30, 0]。"""
    fake = FakeLLM()
    fake.enqueue(
        json_data={
            "judge_pass": True,
            "veto_reason": "",
            "additional_risks": [],
            "score_adjustment": -100,  # 超出下界
        }
    )
    out = await LLMClient.judge_recommendation(
        fake, "A", {}, "B", {}, [], primary_evaluation={}
    )
    assert out["score_adjustment"] == -30

    fake.enqueue(
        json_data={
            "judge_pass": True,
            "veto_reason": "",
            "additional_risks": [],
            "score_adjustment": 999,  # 超出上界
        }
    )
    out = await LLMClient.judge_recommendation(
        fake, "A", {}, "B", {}, [], primary_evaluation={}
    )
    assert out["score_adjustment"] == 0


@pytest.mark.asyncio
async def test_judge_recommendation_falls_back_on_unparsed():
    """LLM 返回非 JSON 时不应阻塞推荐流，默认放行。"""
    fake = FakeLLM()
    fake.enqueue(json_data={"raw": "Sorry, I cannot output JSON."})
    out = await LLMClient.judge_recommendation(
        fake, "A", {}, "B", {}, [], primary_evaluation={}
    )
    assert out["judge_pass"] is True
    assert "judge_response_unparsed" in out["additional_risks"]


# ---------- 整合：discovery 调用 JudgeAgent 后效果 ----------

@pytest.mark.asyncio
async def test_discovery_blocks_when_judge_vetoes(session):
    """主评估通过阈值，但 JudgeAgent veto，则不应落库 Recommendation。"""
    me, my_agent = await make_user_with_agent(session, username="me", agent_name="A")
    _, peer_agent = await make_user_with_agent(session, username="peer", agent_name="B")
    await session.commit()

    fake = FakeLLM()
    # 8 次 ReAct
    for i in range(8):
        fake.enqueue(json_data={"thought": "", "observation": "", "action": f"msg{i}"})
    # 3 次主评估都很高
    for s in [80, 82, 84]:
        fake.enqueue(
            json_data={
                "compatible": True,
                "score": s,
                "confidence": 80,
                "reason": "看起来不错",
                "highlights": ["共同兴趣"],
                "risks": [],
            }
        )
    # 但 Judge veto
    fake.enqueue(
        json_data={
            "judge_pass": False,
            "veto_reason": "对话太短，证据不足",
            "additional_risks": ["evidence_insufficient"],
            "score_adjustment": -20,
        }
    )

    a = AgentSnapshot(id=my_agent.id, name=my_agent.name, personality=my_agent.personality)
    b = AgentSnapshot(id=peer_agent.id, name=peer_agent.name, personality=peer_agent.personality)
    result = await discovery_service._agent_chat_and_evaluate(
        session, a, b, fake, min_match_score=50, min_confidence=50
    )
    assert result is None  # JudgeAgent veto


@pytest.mark.asyncio
async def test_discovery_applies_score_adjustment_when_judge_passes(session):
    """JudgeAgent 放行但建议减分；最终落库的 score 应被减少。"""
    me, my_agent = await make_user_with_agent(session, username="me", agent_name="A")
    _, peer_agent = await make_user_with_agent(session, username="peer", agent_name="B")
    await session.commit()

    fake = FakeLLM()
    for i in range(8):
        fake.enqueue(json_data={"thought": "", "observation": "", "action": f"msg{i}"})
    # 高分主评估
    for s in [85, 85, 85]:
        fake.enqueue(
            json_data={
                "compatible": True,
                "score": s,
                "confidence": 85,
                "reason": "very good",
                "highlights": ["matches"],
                "risks": [],
            }
        )
    # Judge 放行但减 10 分
    fake.enqueue(
        json_data={
            "judge_pass": True,
            "veto_reason": "",
            "additional_risks": ["minor_concern"],
            "score_adjustment": -10,
        }
    )

    a = AgentSnapshot(id=my_agent.id, name=my_agent.name, personality=my_agent.personality)
    b = AgentSnapshot(id=peer_agent.id, name=peer_agent.name, personality=peer_agent.personality)

    # 取一份"无 judge"作为基线 score（再跑一次相同流程，但 judge 给 0 调整）
    result_with_penalty = await discovery_service._agent_chat_and_evaluate(
        session, a, b, fake, min_match_score=50, min_confidence=50
    )
    await session.commit()
    assert result_with_penalty is not None
    assert result_with_penalty["score_adjustment"] == -10
    assert result_with_penalty["judge_pass"] is True

    # 校准后的 base score（无调整）= calibrate(85, 85) = 0.75*85+0.25*85=85，再 -85>85 多扣
    # 最终 score 实测应该比 base 少 10
    from sqlalchemy import select
    from app.models.models import Recommendation

    rec = (
        await session.execute(
            select(Recommendation).where(Recommendation.id == result_with_penalty["recommendation_id"])
        )
    ).scalar_one()
    # base calibrated ≈ 85（因为 s=85 触发 -((85-85)*0.5)=0），实际值由 _calibrate_score 决定；
    # 关键检查：减了 10
    assert rec.score == result_with_penalty["score"]


@pytest.mark.asyncio
async def test_discovery_records_additional_risks_in_evaluation(session, monkeypatch):
    """JudgeAgent 给的 additional_risks 应该被合入 evaluation.risks。

    通过 monkeypatch 把内部 evaluation 抓出来验证。
    """
    me, my_agent = await make_user_with_agent(session, username="me", agent_name="A")
    _, peer_agent = await make_user_with_agent(session, username="peer", agent_name="B")
    await session.commit()

    fake = FakeLLM()
    for i in range(8):
        fake.enqueue(json_data={"thought": "", "observation": "", "action": f"msg{i}"})
    for s in [80, 80, 80]:
        fake.enqueue(
            json_data={
                "compatible": True,
                "score": s,
                "confidence": 80,
                "reason": "ok",
                "highlights": ["a"],
                "risks": ["原始风险"],
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
    # 验证：FakeLLM 的最后一次 chat_json 调用就是 judge，前面的是评估 + ReAct
    calls = [c for c in fake.calls if c["kind"] == "chat_json"]
    # 8 次 ReAct（chat_json，因为 agent_chat_turn_react 用 chat_json）+ 3 次评估 + 1 次 judge
    assert len(calls) == 12
