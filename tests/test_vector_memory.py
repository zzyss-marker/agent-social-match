"""向量语义召回测试。"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.core.config import Settings
from app.services import discovery_service
from app.services.discovery_service import (
    _rule_overlap_score,
    hybrid_recall_score,
)
from app.services.embedding_service import (
    cosine_similarity,
    embed_text,
    ensure_agent_embedding,
    personality_to_text,
    strip_internal_fields,
    text_hash,
    _local_bigram_vector,
)
from tests.conftest import make_user_with_agent


def _bare_settings() -> Settings:
    """造一个不连接外部 embedding endpoint 的 Settings，让 fallback 生效。"""
    s = Settings(SESSION_SECRET="x" * 64, EMAIL_CODE_SECRET="y" * 64)
    s.LLM_BASE_URL = ""
    s.LLM_API_KEY = ""
    return s


# ---------- personality_to_text / strip_internal_fields ----------

def test_personality_to_text_concatenates_relevant_fields():
    profile = {
        "traits": ["温柔", "细心"],
        "interests": ["旅行"],
        "looking_for": "灵魂伴侣",
        "vibe": "温暖治愈",
        "context_memory": ["在上海读书"],
        "boundaries": ["不接受异地"],  # 不应进入向量化文本
    }
    text = personality_to_text(profile)
    assert "温柔" in text
    assert "旅行" in text
    assert "灵魂伴侣" in text
    assert "温暖治愈" in text
    assert "在上海读书" in text
    assert "不接受异地" not in text  # boundaries 不参与向量化


def test_strip_internal_fields_removes_underscored_keys():
    p = {
        "traits": ["x"],
        "_embedding_vector": [0.1, 0.2],
        "_embedding_hash": "abc",
        "interests": ["y"],
    }
    cleaned = strip_internal_fields(p)
    assert "_embedding_vector" not in cleaned
    assert "_embedding_hash" not in cleaned
    assert cleaned["traits"] == ["x"]
    assert cleaned["interests"] == ["y"]


# ---------- 本地 bigram 向量化 ----------

def test_local_bigram_vector_normalized():
    v = _local_bigram_vector("健身房")
    norm_sq = sum(x * x for x in v)
    assert abs(norm_sq - 1.0) < 1e-6 or norm_sq == 0.0
    assert len(v) == 256


def test_cosine_similarity_finds_lexical_neighbors():
    """'健身房' 和 '健身' 共享 bigram '健身'，相似度应该 > '健身' 和 '阅读'。"""
    v1 = _local_bigram_vector("健身房")
    v2 = _local_bigram_vector("健身")
    v3 = _local_bigram_vector("阅读电影")
    sim_close = cosine_similarity(v1, v2)
    sim_far = cosine_similarity(v1, v3)
    assert sim_close > sim_far
    assert sim_close > 0.3  # 共享 bigram 的实质相似


def test_cosine_similarity_handles_empty():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0], []) == 0.0
    assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0  # 维度不匹配


# ---------- embed_text fallback ----------

@pytest.mark.asyncio
async def test_embed_text_falls_back_to_local_when_no_endpoint():
    settings = _bare_settings()
    vec = await embed_text("旅行 摄影 咖啡", settings)
    assert isinstance(vec, list)
    assert len(vec) > 0
    # 一致性：同样输入要稳定输出
    vec2 = await embed_text("旅行 摄影 咖啡", settings)
    assert vec == vec2


@pytest.mark.asyncio
async def test_embed_text_empty_input():
    settings = _bare_settings()
    assert await embed_text("", settings) == []
    assert await embed_text("   ", settings) == []


# ---------- ensure_agent_embedding 缓存 ----------

@pytest.mark.asyncio
async def test_ensure_agent_embedding_caches_in_personality(session):
    settings = _bare_settings()
    _, agent = await make_user_with_agent(
        session, username="me", interests=["旅行", "摄影"]
    )
    await session.commit()

    vec1 = await ensure_agent_embedding(session, agent, settings)
    assert len(vec1) > 0
    # 第二次调用应该命中缓存（personality 没变）
    profile_after = dict(agent.personality)
    assert profile_after.get("_embedding_hash")
    assert profile_after.get("_embedding_vector") == vec1


@pytest.mark.asyncio
async def test_ensure_agent_embedding_regenerates_on_text_change(session):
    settings = _bare_settings()
    _, agent = await make_user_with_agent(session, username="me", interests=["旅行"])
    await session.commit()

    vec1 = await ensure_agent_embedding(session, agent, settings)
    h1 = agent.personality.get("_embedding_hash")

    # 修改画像
    profile = dict(agent.personality)
    profile["interests"] = ["旅行", "潜水"]
    agent.personality = profile
    await session.flush()

    vec2 = await ensure_agent_embedding(session, agent, settings)
    h2 = agent.personality.get("_embedding_hash")
    assert h1 != h2
    assert vec1 != vec2


# ---------- 混合召回融合 ----------

def test_hybrid_recall_score_weighting():
    # 纯向量满分 + 规则 0
    s1 = hybrid_recall_score(rule_score=0.0, vector_similarity=1.0)
    # 纯规则中等 + 向量 0
    s2 = hybrid_recall_score(rule_score=3.0, vector_similarity=0.0)
    # 混合
    s3 = hybrid_recall_score(rule_score=3.0, vector_similarity=1.0)
    assert s3 > s1
    assert s3 > s2
    assert 0.0 <= s1 <= 1.0
    assert 0.0 <= s2 <= 1.0


def test_rule_overlap_score_unchanged_for_legacy_path():
    a = {"traits": ["温柔"], "interests": ["旅行"], "looking_for": "灵魂伴侣", "vibe": "温暖"}
    b = {"traits": ["温柔"], "interests": ["旅行"], "looking_for": "灵魂伴侣", "vibe": "温暖"}
    assert _rule_overlap_score(a, b) == 1.8 + 1.2 + 1.2 + 0.8


# ---------- 端到端：discovery prefilter 优先选高语义相似 ----------

@pytest.mark.asyncio
async def test_discovery_prefers_semantic_neighbor_over_unrelated(session):
    """me=['健身房'] 的高分应该是 ['健身','跑步']，而不是 ['阅读']。"""
    settings = _bare_settings()
    me, my_agent = await make_user_with_agent(
        session, username="me", agent_name="Me",
        interests=["健身房"], traits=["热爱运动"]
    )
    _, fitness = await make_user_with_agent(
        session, username="fit", agent_name="健身搭子",
        interests=["健身", "跑步"], traits=["热爱运动"]
    )
    _, reader = await make_user_with_agent(
        session, username="rd", agent_name="阅读者",
        interests=["阅读", "电影"], traits=["内向"]
    )
    await session.commit()

    # 模拟 run_discovery 召回部分的核心计算
    my_vec = await ensure_agent_embedding(session, my_agent, settings)
    fitness_vec = await ensure_agent_embedding(session, fitness, settings)
    reader_vec = await ensure_agent_embedding(session, reader, settings)

    fit_hybrid = hybrid_recall_score(
        _rule_overlap_score(my_agent.personality, fitness.personality),
        cosine_similarity(my_vec, fitness_vec),
    )
    rd_hybrid = hybrid_recall_score(
        _rule_overlap_score(my_agent.personality, reader.personality),
        cosine_similarity(my_vec, reader_vec),
    )
    # 健身候选应该比阅读候选分数高
    assert fit_hybrid > rd_hybrid


@pytest.mark.asyncio
async def test_personality_internal_fields_stripped_from_snapshot(session):
    settings = _bare_settings()
    _, agent = await make_user_with_agent(session, username="me", interests=["旅行"])
    await ensure_agent_embedding(session, agent, settings)
    await session.commit()

    # 模拟 discovery 中创建 snapshot
    snap = discovery_service.AgentSnapshot(
        id=agent.id,
        name=agent.name,
        personality=strip_internal_fields(agent.personality),
    )
    assert "_embedding_vector" not in snap.personality
    assert "_embedding_hash" not in snap.personality
    assert "interests" in snap.personality
