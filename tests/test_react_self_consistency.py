"""测试 ReAct（Yao 2022） + Self-Consistency（Wang 2022）改造。

- agent_chat_turn_react: 输出结构化 thought/action/observation
- evaluate_match_self_consistent: 多次采样取中位数
- _agent_chat_and_evaluate: 整合 ReAct + Self-Consistency 后端到端落库
"""
from __future__ import annotations

import json

import pytest

from app.services import discovery_service
from app.services.discovery_service import AgentSnapshot
from app.services.llm_client import LLMClient
from tests.conftest import FakeLLM, make_user_with_agent


# ---------- agent_chat_turn_react ----------

@pytest.mark.asyncio
async def test_agent_chat_turn_react_parses_structured_output():
    fake = FakeLLM()
    fake.enqueue(
        json_data={
            "thought": "对方提到旅行，我可以分享共鸣",
            "observation": "对方很外向",
            "action": "我也喜欢旅行，最近去了云南。",
        }
    )
    react = await LLMClient.agent_chat_turn_react(
        fake,
        "Aria",
        {"traits": ["外向"], "interests": ["旅行"]},
        [{"role": "user", "content": "Bob: 我喜欢旅行"}],
    )
    assert react["action"] == "我也喜欢旅行，最近去了云南。"
    assert react["thought"].startswith("对方提到")
    assert react["observation"] == "对方很外向"


@pytest.mark.asyncio
async def test_agent_chat_turn_react_falls_back_on_raw_output():
    """模型直接给纯文本而不是 JSON 时，要把文本作为 action 兜底。"""
    fake = FakeLLM()
    # FakeLLM.chat_json 直接返回 json_data；模拟解析失败用 {"raw": "..."}
    fake.enqueue(json_data={"raw": "我也喜欢旅行。"})
    react = await LLMClient.agent_chat_turn_react(
        fake, "X", {}, [{"role": "user", "content": "hi"}]
    )
    assert react["action"] == "我也喜欢旅行。"
    assert react["thought"] == ""


# ---------- Self-Consistency ----------

@pytest.mark.asyncio
async def test_evaluate_match_self_consistent_takes_median_score():
    fake = FakeLLM()
    # 三次采样：score 60 / 80 / 70 → 中位数 70
    fake.enqueue(
        json_data={
            "compatible": True,
            "score": 60,
            "confidence": 70,
            "reason": "样本一",
            "highlights": ["h1"],
            "risks": [],
        }
    )
    fake.enqueue(
        json_data={
            "compatible": True,
            "score": 80,
            "confidence": 80,
            "reason": "样本二",
            "highlights": ["h2"],
            "risks": [],
        }
    )
    fake.enqueue(
        json_data={
            "compatible": True,
            "score": 70,
            "confidence": 75,
            "reason": "样本三",
            "highlights": ["h3"],
            "risks": ["r3"],
        }
    )

    out = await LLMClient.evaluate_match_self_consistent(
        fake,
        "A", {"traits": ["a"]},
        "B", {"traits": ["b"]},
        [{"role": "user", "content": "hi"}],
        samples=3,
    )
    assert out["score"] == 70
    assert out["confidence"] == 75
    assert out["compatible"] is True
    # anchor 是中位 score 对应的样本（70），所以 reason / highlights 来自样本三
    assert out["reason"] == "样本三"
    assert "h3" in out["highlights"]
    assert "r3" in out["risks"]
    assert len(out["sampled"]) == 3


@pytest.mark.asyncio
async def test_evaluate_match_self_consistent_majority_vote_on_compatible():
    """compatible 字段按多数投票决定。2 个 false + 1 个 true → false。"""
    fake = FakeLLM()
    fake.enqueue(json_data={"compatible": False, "score": 40, "confidence": 50, "reason": "a", "highlights": [], "risks": []})
    fake.enqueue(json_data={"compatible": False, "score": 45, "confidence": 60, "reason": "b", "highlights": [], "risks": []})
    fake.enqueue(json_data={"compatible": True, "score": 90, "confidence": 80, "reason": "c", "highlights": [], "risks": []})
    out = await LLMClient.evaluate_match_self_consistent(
        fake,
        "A", {}, "B", {},
        [],
        samples=3,
    )
    assert out["compatible"] is False
    # 中位 score = 45
    assert out["score"] == 45


@pytest.mark.asyncio
async def test_evaluate_match_self_consistent_handles_all_failures():
    fake = FakeLLM()
    # 所有采样都返回非 dict（FakeLLM 只在 enqueue 时给 json_data，
    # 未 enqueue 时返回 {} 但 evaluate_match 包装层最终把 {} 视为合法 dict）
    # 我们用空 dict 模拟 LLM 总是输出非结构化结果
    for _ in range(3):
        fake.enqueue(json_data={})
    out = await LLMClient.evaluate_match_self_consistent(
        fake,
        "A", {}, "B", {},
        [],
        samples=3,
    )
    # score/confidence 都是 0；compatible 默认 False
    assert out["score"] == 0
    assert out["confidence"] == 0
    assert out["compatible"] is False


# ---------- 整合：_agent_chat_and_evaluate ----------

@pytest.mark.asyncio
async def test_agent_chat_and_evaluate_uses_react_and_self_consistency(session):
    """端到端：8 次 ReAct 对话 + 3 次评估采样后落库 Recommendation。"""
    me, my_agent = await make_user_with_agent(
        session, username="me", agent_name="Aria", interests=["旅行", "摄影"]
    )
    _, peer_agent = await make_user_with_agent(
        session, username="peer", agent_name="云", interests=["旅行", "潜水"]
    )
    await session.commit()

    fake = FakeLLM()
    # 8 次 ReAct 调用（4 轮，每轮各方各 1 次）
    for i in range(8):
        fake.enqueue(
            json_data={
                "thought": f"思考{i}",
                "observation": f"观察{i}",
                "action": f"对话内容{i}",
            }
        )
    # 3 次评估采样：score 70/72/74 → median 72；calibrated 大约 (0.75*72)+(0.25*75) ≈ 72.75
    for s, c in [(70, 70), (72, 75), (74, 80)]:
        fake.enqueue(
            json_data={
                "compatible": True,
                "score": s,
                "confidence": c,
                "reason": f"score={s}",
                "highlights": ["共同兴趣旅行"],
                "risks": [],
            }
        )

    a = AgentSnapshot(id=my_agent.id, name=my_agent.name, personality=my_agent.personality)
    b = AgentSnapshot(id=peer_agent.id, name=peer_agent.name, personality=peer_agent.personality)

    result = await discovery_service._agent_chat_and_evaluate(
        session, a, b, fake, min_match_score=50, min_confidence=50
    )
    await session.commit()

    assert result is not None
    assert result["recommendation_id"] is not None
    assert result["react_turns"] == 8  # 4 轮 × 2 人
    assert result["samples"] == 3

    # 验证 Recommendation 落库且 reason 与 anchor 一致
    from sqlalchemy import select
    from app.models.models import Recommendation

    rec = (
        await session.execute(select(Recommendation).where(Recommendation.id == result["recommendation_id"]))
    ).scalar_one()
    assert rec.reason == "score=72"  # anchor 是中位 score 对应的


@pytest.mark.asyncio
async def test_agent_chat_and_evaluate_blocks_low_confidence(session):
    """confidence 不足时不应该落库 Recommendation。"""
    me, my_agent = await make_user_with_agent(session, username="me", agent_name="Aria")
    _, peer_agent = await make_user_with_agent(session, username="peer", agent_name="P")
    await session.commit()

    fake = FakeLLM()
    for i in range(8):
        fake.enqueue(json_data={"thought": "", "observation": "", "action": f"msg{i}"})
    for s in [80, 82, 84]:
        fake.enqueue(
            json_data={
                "compatible": True,
                "score": s,
                "confidence": 30,  # 低 confidence
                "reason": "x",
                "highlights": [],
                "risks": [],
            }
        )

    a = AgentSnapshot(id=my_agent.id, name=my_agent.name, personality=my_agent.personality)
    b = AgentSnapshot(id=peer_agent.id, name=peer_agent.name, personality=peer_agent.personality)
    result = await discovery_service._agent_chat_and_evaluate(
        session, a, b, fake, min_match_score=50, min_confidence=55
    )
    assert result is None
