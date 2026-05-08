"""画像语义去重 + MMR 多样性 + chat_with_agent 速率限制 测试。"""
from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select

from app.main import _bigrams, _dedupe_semantic, _is_semantic_dup, _merge_profile
from app.models.models import Agent, DiscoveryAttempt
from app.services.discovery_service import (
    _epsilon_greedy_swap,
    _fuzzy_overlap_count,
    _mmr_select,
    _rule_overlap_score,
)
from tests.conftest import make_user_with_agent


# =============== 画像语义去重 ===============


class _FakeAgent:
    def __init__(self, agent_id: int) -> None:
        self.id = agent_id


def test_dedupe_collapses_user_actual_duplicates():
    """用户实际遇到的重复条目（蛋仔派对的 6 个变体）必须收敛到 1 条。"""
    raw = [
        "蛋仔派对",
        "乐园模式游戏",
        "乐园模式",
        "自由创造",
        "游戏乐园模式",
        "蛋仔派对乐园模式",
    ]
    out = _dedupe_semantic(raw)
    assert len(out) <= 2, f"expected ≤ 2 distinct concepts, got {out}"
    assert any("蛋仔" in x for x in out), out


def test_substring_containment_keeps_most_specific():
    items = ["乐园模式", "蛋仔派对乐园模式"]
    out = _dedupe_semantic(items)
    assert out == ["蛋仔派对乐园模式"]


def test_word_order_variant_is_dedup():
    a, b = "游戏乐园模式", "乐园模式游戏"
    assert _is_semantic_dup(a, b)


def test_distinct_interests_are_kept():
    out = _dedupe_semantic(["运动", "读书", "音乐", "旅行"])
    assert out == ["运动", "读书", "音乐", "旅行"]


def test_bigrams_simple():
    assert _bigrams("乐园模式") == {"乐园", "园模", "模式"}
    assert _bigrams("a") == {"a"}
    assert _bigrams("") == set()


def test_merge_profile_dedupes_extracted_duplicates():
    old = {"interests": ["蛋仔派对"]}
    new = {"interests": ["蛋仔派对乐园模式", "乐园模式游戏", "游戏乐园模式"]}
    merged = _merge_profile(old, new)
    # 三个变体 + 一个旧条目，应该收敛到 1（最具体）
    assert len(merged["interests"]) == 1
    assert "蛋仔派对" in merged["interests"][0]


# =============== rule_overlap_score 模糊匹配 ===============


def test_fuzzy_overlap_handles_partial_token_share():
    """'蛋仔派对' 和 '派对游戏' 共享 token '派对'，应该有正向分数。"""
    score = _fuzzy_overlap_count({"蛋仔派对"}, {"派对游戏"})
    assert score > 0


def test_rule_overlap_score_uses_fuzzy_match():
    a = {"interests": ["蛋仔派对"]}
    b = {"interests": ["派对游戏"]}
    score_fuzzy = _rule_overlap_score(a, b)
    # 改造前 set 交集为 0，改造后通过 bigram fuzzy 应该 > 0
    assert score_fuzzy > 0


# =============== MMR 多样性重排 ===============


def test_mmr_picks_relevant_then_diverse():
    """MMR 第一个挑相关度最高的，后续偏好不太相似的。"""
    agents = [_FakeAgent(i) for i in range(5)]
    # rel 排序：a0 > a1 > a2 > a3 > a4
    # 但 a0 和 a1 向量几乎一样；a0 和 a2 向量正交
    candidates = [
        (0.9, agents[0], [1.0, 0.0]),
        (0.85, agents[1], [0.99, 0.05]),  # 与 a0 几乎重复
        (0.7, agents[2], [0.0, 1.0]),     # 与 a0 完全不同
        (0.6, agents[3], [0.5, 0.5]),
        (0.5, agents[4], [-1.0, 0.0]),    # 反向
    ]
    picks = _mmr_select(candidates, k=3, lambda_=0.5)
    pick_ids = [a.id for a in picks]
    assert pick_ids[0] == 0  # 第一个总是相关度最高
    # 第二个不应是 a1（太像 a0），应跳到 a2 或 a4
    assert pick_ids[1] != 1


def test_mmr_returns_all_when_k_exceeds_pool():
    agents = [_FakeAgent(i) for i in range(2)]
    candidates = [(0.5, agents[0], [1.0]), (0.4, agents[1], [0.0])]
    picks = _mmr_select(candidates, k=10, lambda_=0.7)
    assert len(picks) == 2


def test_epsilon_greedy_no_swap_when_epsilon_zero():
    agents = [_FakeAgent(i) for i in range(5)]
    mmr = agents[:3]
    out = _epsilon_greedy_swap(mmr, agents, epsilon=0.0)
    assert [a.id for a in out] == [0, 1, 2]


def test_epsilon_greedy_swaps_when_epsilon_one():
    agents = [_FakeAgent(i) for i in range(5)]
    mmr = agents[:3]
    out = _epsilon_greedy_swap(mmr, agents, epsilon=1.0)
    # 必然替换最后一个；新尾部应来自 outsiders {3, 4}
    assert out[-1].id in (3, 4)
    assert [a.id for a in out[:2]] == [0, 1]


# =============== chat_with_agent 速率限制 ===============


@pytest.mark.asyncio
async def test_chat_with_agent_directed_limit(session, monkeypatch):
    """每日定向次数用尽后应当返回 ok=False 并提示。"""
    from app.core.config import Settings
    from app.services.agent_tools import chat_with_agent

    me, my_agent = await make_user_with_agent(session, username="me", agent_name="A")
    other_user, other_agent = await make_user_with_agent(session, username="o", agent_name="木木")
    await session.commit()

    settings = Settings(SESSION_SECRET="x" * 64, EMAIL_CODE_SECRET="y" * 64,
                        SECURITY_REQUIRE_STRONG_SECRETS=False)
    settings.DISCOVERY_DIRECTED_DAILY_LIMIT = 1

    # 预先塞一条今日的定向 attempt → 已用尽
    session.add(DiscoveryAttempt(
        from_agent_id=my_agent.id,
        to_agent_id=other_agent.id,
        produced_rec=False,
        score=None,
        source="directed",
    ))
    await session.commit()

    class _FakeLLM:
        pass

    res = await chat_with_agent(
        session=session,
        current_user_id=me.id,
        target="木木",
        llm=_FakeLLM(),
        settings=settings,
    )
    assert res["ok"] is False
    assert "次数" in res["message"] or "limit" in res["message"].lower()
