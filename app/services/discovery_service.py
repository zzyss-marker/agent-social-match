from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.db_gate import GatePriority, write_gate
from app.core.time_utils import ensure_utc8, now_utc8
from app.models.models import (
    Agent,
    Conversation,
    ConversationParticipant,
    DiscoveryAttempt,
    Message,
    Recommendation,
)
from app.schemas.schemas import DiscoveryResponse
from app.services.embedding_service import (
    cosine_similarity,
    ensure_agent_embedding,
    personality_to_text,
    strip_internal_fields,
)
from app.services.llm_client import LLMClient

_DISCOVERY_LOCKS: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


@dataclass(slots=True)
class AgentSnapshot:
    id: int
    name: str
    personality: dict[str, Any]


def _is_sqlite_locked_error(exc: Exception) -> bool:
    return "database is locked" in str(exc).lower()


def _to_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(v).strip().lower() for v in value if str(v).strip()}


def _bigrams_lower(s: str) -> set[str]:
    s = str(s).strip().lower()
    if len(s) <= 1:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _fuzzy_overlap_count(a_items: set[str], b_items: set[str]) -> float:
    """Token-bigram 模糊重合计数（containment ratio）。

    对每个 a 中的条目，找到 b 里最接近的（bigram 含 ratio = inter / min），
    把含 ratio（≥0.3 才计入）累加。这样 "蛋仔派对" / "派对游戏" 共享一个
    token "派对" 时也会贡献正向信号（containment ≈ 0.33），解决 set 严格
    交集为 0 导致的"看似合理却不被召回"问题。
    """
    if not a_items or not b_items:
        return 0.0
    b_bigrams = [(item, _bigrams_lower(item)) for item in b_items]
    total = 0.0
    for a_item in a_items:
        ag = _bigrams_lower(a_item)
        if not ag:
            continue
        best = 0.0
        for _b_item, bg in b_bigrams:
            if not bg:
                continue
            inter = len(ag & bg)
            smaller = min(len(ag), len(bg))
            if smaller == 0:
                continue
            containment = inter / smaller
            if containment > best:
                best = containment
        if best >= 0.3:
            total += best
    return total


def _prefilter_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Cheap retrieval score to reduce LLM calls when agent pool is large.

    现在等价于 _rule_overlap_score —— 保留作为单独的规则信号，
    与向量 cosine 相似度在召回阶段加权融合。
    """
    return _rule_overlap_score(a, b)


def _rule_overlap_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    """规则信号：兴趣/特征/looking_for/vibe 的混合重合度。

    集合精确交集 + bigram Jaccard 模糊重合 → 让 "蛋仔派对" 与 "派对游戏" 也能部分 match。
    """
    a_traits = _to_set(a.get("traits"))
    b_traits = _to_set(b.get("traits"))
    a_interests = _to_set(a.get("interests"))
    b_interests = _to_set(b.get("interests"))
    a_looking_for = str(a.get("looking_for", "")).strip()
    b_looking_for = str(b.get("looking_for", "")).strip()
    a_vibe = str(a.get("vibe", "")).strip()
    b_vibe = str(b.get("vibe", "")).strip()

    # 精确交集（强信号）
    trait_exact = len(a_traits & b_traits)
    interest_exact = len(a_interests & b_interests)
    # 模糊匹配（弱信号，但能补救字面不同的同义词）
    interest_fuzzy = _fuzzy_overlap_count(a_interests - b_interests, b_interests - a_interests)
    trait_fuzzy = _fuzzy_overlap_count(a_traits - b_traits, b_traits - a_traits)

    looking_bonus = 1 if a_looking_for and b_looking_for and a_looking_for == b_looking_for else 0
    vibe_bonus = 1 if a_vibe and b_vibe and a_vibe == b_vibe else 0

    return (
        interest_exact * 1.8
        + trait_exact * 1.2
        + interest_fuzzy * 0.9
        + trait_fuzzy * 0.6
        + looking_bonus * 1.2
        + vibe_bonus * 0.8
    )


def hybrid_recall_score(
    rule_score: float,
    vector_similarity: float,
    *,
    rule_weight: float = 0.4,
    vector_weight: float = 0.6,
) -> float:
    """规则信号 + 向量 cosine 加权融合。

    规则信号的取值大致 [0, ~6]，先压到 [0, 1]：min(rule_score / 6, 1)
    向量 cosine 取值 [-1, 1]，先压到 [0, 1]：(cos + 1) / 2
    """
    rule_norm = max(0.0, min(rule_score / 6.0, 1.0))
    vec_norm = max(0.0, min((vector_similarity + 1.0) / 2.0, 1.0))
    return rule_weight * rule_norm + vector_weight * vec_norm


def _calibrate_score(raw_score: int, confidence: int) -> int:
    """Conservative score calibration to reduce optimistic match inflation."""
    s = max(0, min(100, int(raw_score)))
    c = max(0, min(100, int(confidence)))
    calibrated = (0.75 * s) + (0.25 * c)
    if s > 85:
        calibrated -= (s - 85) * 0.5
    if c < 60:
        calibrated -= (60 - c) * 0.6
    return max(0, min(100, int(round(calibrated))))


def _as_aware_utc8(dt: datetime | None) -> datetime:
    localized = ensure_utc8(dt)
    if localized is None:
        return now_utc8()
    return localized

async def _load_blocked_targets(
    session: AsyncSession,
    agent_id: int,
    cooldown_hours: int,
    attempt_cooldown_hours: int,
) -> set[int]:
    """两层冷却：

    1) Recommendation 表：pending 状态永久屏蔽，已产生过 rec 的 pair 在 cooldown_hours 内屏蔽
    2) DiscoveryAttempt 表：评估过但没生成 rec 的 pair 在 attempt_cooldown_hours 内短冷却
       （避免同一 top-1 候选每次都被选、每次都不够分，死循环卡住）
    """
    rec_since = now_utc8() - timedelta(hours=max(1, cooldown_hours))
    attempt_since = now_utc8() - timedelta(hours=max(1, attempt_cooldown_hours))

    blocked: set[int] = set()

    # 1) Recommendation 屏蔽
    rows = (
        await session.execute(
            select(
                Recommendation.from_agent_id,
                Recommendation.to_agent_id,
                Recommendation.status,
                Recommendation.created_at,
            ).where(
                or_(
                    Recommendation.from_agent_id == agent_id,
                    Recommendation.to_agent_id == agent_id,
                )
            )
        )
    ).all()
    for from_id, to_id, status, created_at in rows:
        peer_id = to_id if from_id == agent_id else from_id
        created = _as_aware_utc8(created_at)
        if status == "pending" or created >= rec_since:
            blocked.add(int(peer_id))

    # 2) DiscoveryAttempt 短冷却（不论是否产生 rec，最近评估过都短期屏蔽）
    attempt_rows = (
        await session.execute(
            select(
                DiscoveryAttempt.from_agent_id,
                DiscoveryAttempt.to_agent_id,
                DiscoveryAttempt.created_at,
            ).where(
                or_(
                    DiscoveryAttempt.from_agent_id == agent_id,
                    DiscoveryAttempt.to_agent_id == agent_id,
                )
            )
        )
    ).all()
    for from_id, to_id, created_at in attempt_rows:
        peer_id = to_id if from_id == agent_id else from_id
        created = _as_aware_utc8(created_at)
        if created >= attempt_since:
            blocked.add(int(peer_id))

    return blocked


async def _pending_recommendation_count(session: AsyncSession, my_agent_id: int) -> int:
    result = await session.execute(
        select(func.count(Recommendation.id)).where(
            and_(
                or_(
                    Recommendation.from_agent_id == my_agent_id,
                    Recommendation.to_agent_id == my_agent_id,
                ),
                Recommendation.status == "pending",
            )
        )
    )
    return int(result.scalar() or 0)


def _mmr_select(
    candidates: list[tuple[float, Agent, list[float]]],
    k: int,
    lambda_: float = 0.7,
) -> list[Agent]:
    """Maximal Marginal Relevance 重排：相关度 + 候选间相似度惩罚。

    Carbonell & Goldstein 1998. 公式：
      MMR = argmax_i [ λ · Rel(i) - (1-λ) · max_{j∈selected} Sim(i, j) ]

    candidates: list of (relevance_score, Agent, embedding_vector)
    每次贪心选下一个：在剩余候选里挑 MMR 分最高的，直到选满 k 个。
    """
    if not candidates:
        return []
    if k >= len(candidates):
        return [c[1] for c in candidates]

    # 已选 / 剩余
    selected: list[tuple[float, Agent, list[float]]] = []
    remaining = list(candidates)

    # 先选相关度最高的（max_sim 项为 0）
    remaining.sort(key=lambda x: x[0], reverse=True)
    selected.append(remaining.pop(0))

    while remaining and len(selected) < k:
        best_idx = 0
        best_score = -1e9
        for i, (rel, _agent, vec) in enumerate(remaining):
            max_sim = 0.0
            if vec:
                for _, _sa, sv in selected:
                    if not sv:
                        continue
                    sim = cosine_similarity(vec, sv)
                    if sim > max_sim:
                        max_sim = sim
            score = lambda_ * rel - (1.0 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return [c[1] for c in selected]


def _epsilon_greedy_swap(
    mmr_picks: list[Agent],
    full_pool: list[Agent],
    epsilon: float,
) -> list[Agent]:
    """在 MMR 结果之上做 ε-greedy 探索：

    以概率 ε 把最后一个 MMR 候选替换为"全池中未被选中的随机一个"。
    这是 SMMR (Sampling-based MMR) 的简化版，让"分数较低但可能合得来"的人也有机会被探索。
    """
    if not mmr_picks or epsilon <= 0:
        return mmr_picks
    if random.random() >= epsilon:
        return mmr_picks
    picked_ids = {a.id for a in mmr_picks}
    outsiders = [a for a in full_pool if a.id not in picked_ids]
    if not outsiders:
        return mmr_picks
    swap_in = random.choice(outsiders)
    return mmr_picks[:-1] + [swap_in]


async def _record_attempt(
    session: AsyncSession,
    from_agent_id: int,
    to_agent_id: int,
    produced_rec: bool,
    score: int | None,
    source: str = "auto",
) -> None:
    """落一条 DiscoveryAttempt 记录，让短冷却生效。"""
    session.add(
        DiscoveryAttempt(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            produced_rec=produced_rec,
            score=score,
            source=source,
        )
    )


async def run_discovery(
    session: AsyncSession,
    agent_id: int,
    llm: LLMClient,
    settings: Settings,
) -> DiscoveryResponse:
    """Have an agent discover and chat with other agents."""
    lock = _DISCOVERY_LOCKS[agent_id]
    async with lock:
        result = await session.execute(select(Agent).where(Agent.id == agent_id))
        my_agent = result.scalar_one_or_none()
        if my_agent is None:
            raise ValueError("Agent not found")

        pending_count = await _pending_recommendation_count(session, my_agent.id)
        if pending_count >= settings.DISCOVERY_MAX_PENDING_RECOMMENDATIONS:
            my_agent.status = "idle"
            await session.flush()
            return DiscoveryResponse(
                new_recommendations=0,
                details=[f"待处理推荐过多（{pending_count}条），请先处理后再探索。"],
            )

        result = await session.execute(select(Agent).where(Agent.id != agent_id))
        candidates = result.scalars().all()
        if not candidates:
            my_agent.status = "idle"
            await session.flush()
            return DiscoveryResponse(new_recommendations=0, details=["社区里还没有其他 Agent"])

        my_snapshot = AgentSnapshot(
            id=my_agent.id,
            name=my_agent.name,
            personality=strip_internal_fields(my_agent.personality or {}),
        )

        # 确保我自己的向量已生成
        my_vector = await ensure_agent_embedding(session, my_agent, settings)

        blocked_targets = await _load_blocked_targets(
            session=session,
            agent_id=my_snapshot.id,
            cooldown_hours=settings.DISCOVERY_REC_COOLDOWN_HOURS,
            attempt_cooldown_hours=settings.DISCOVERY_ATTEMPT_COOLDOWN_HOURS,
        )

        # scored: (relevance_score, Agent, embedding_vector) — embedding 留作 MMR 重排用
        scored: list[tuple[float, Agent, list[float]]] = []
        for candidate in candidates:
            if candidate.id in blocked_targets:
                continue
            # 1) 规则重合度（精确交集 + bigram Jaccard 模糊匹配）
            rule_score = _rule_overlap_score(my_snapshot.personality, candidate.personality or {})
            # 2) 向量 cosine（候选向量按需生成）
            cand_vec: list[float] = []
            try:
                cand_vec = await ensure_agent_embedding(session, candidate, settings) or []
                cos = cosine_similarity(my_vector, cand_vec) if my_vector and cand_vec else 0.0
            except Exception:
                cos = 0.0
            score = hybrid_recall_score(rule_score, cos)
            scored.append((score, candidate, cand_vec))

        if not scored:
            my_agent.status = "idle"
            await session.flush()
            return DiscoveryResponse(new_recommendations=0, details=["近期已探索过主要候选，稍后再试"])

        scored.sort(key=lambda item: item[0], reverse=True)
        candidate_pool_limit = max(1, settings.DISCOVERY_CANDIDATE_POOL_LIMIT)
        candidate_pool = scored[:candidate_pool_limit]

        min_chat = max(1, settings.DISCOVERY_CHAT_MIN_PER_RUN)
        max_chat = max(min_chat, settings.DISCOVERY_CHAT_MAX_PER_RUN)
        num_to_chat = min(random.randint(min_chat, max_chat), len(candidate_pool))

        # MMR 重排：在 top-K 池中权衡相关度 vs 候选间多样性，避免每次都挑最像的同一拨人
        mmr_picks = _mmr_select(
            candidate_pool,
            k=num_to_chat,
            lambda_=float(settings.DISCOVERY_MMR_LAMBDA),
        )
        # ε-greedy 探索：偶尔跳出 MMR 头部，把"分数稍低但可能合得来"的候选也试一下
        full_pool_agents = [c[1] for c in candidate_pool]
        selected = _epsilon_greedy_swap(
            mmr_picks,
            full_pool_agents,
            epsilon=float(settings.DISCOVERY_EPSILON),
        )

        # Materialize snapshots before commits/rollbacks to avoid expired ORM lazy-load issues.
        selected_snapshots = [
            AgentSnapshot(
                id=int(t.id),
                name=str(t.name),
                personality=strip_internal_fields(t.personality or {}),
            )
            for t in selected
        ]

        details: list[str] = []
        new_recs = 0

        for target_snapshot in selected_snapshots:
            target_name = target_snapshot.name
            try:
                rec = await _agent_chat_and_evaluate(
                    session=session,
                    agent_a=my_snapshot,
                    agent_b=target_snapshot,
                    llm=llm,
                    min_match_score=max(0, settings.DISCOVERY_MIN_MATCH_SCORE),
                    min_confidence=max(0, settings.DISCOVERY_MIN_CONFIDENCE),
                )
                # 不论是否产生 rec，都记一条 attempt 用于短冷却
                await _record_attempt(
                    session,
                    from_agent_id=my_snapshot.id,
                    to_agent_id=target_snapshot.id,
                    produced_rec=bool(rec),
                    score=rec["score"] if rec else None,
                    source="auto",
                )
                # 探索提交走门闸 DISCOVERY 优先级，用户对话（CHAT）永远优先
                async with write_gate(
                    GatePriority.DISCOVERY,
                    timeout=settings.DB_GATE_DEFAULT_TIMEOUT_SECONDS,
                    label="discovery_run_commit",
                ):
                    await session.commit()
                if rec:
                    new_recs += 1
                    details.append(
                        f"与 {target_name} 对话后通过筛选（分数 {rec['score']}，置信度 {rec['confidence']}）。"
                    )
                else:
                    details.append(f"与 {target_name} 对话后未达到推荐阈值。")
            except Exception as exc:
                await session.rollback()
                if _is_sqlite_locked_error(exc):
                    details.append(f"与 {target_name} 处理失败：数据库繁忙，请稍后重试。")
                else:
                    details.append(f"与 {target_name} 对话出错：{str(exc)[:80]}")

        my_agent.status = "idle"
        await session.flush()

        return DiscoveryResponse(new_recommendations=new_recs, details=details)


async def run_directed_discovery(
    session: AsyncSession,
    agent_id: int,
    target_query: str,
    llm: LLMClient,
    settings: Settings,
) -> dict[str, Any]:
    """主动定向 1:1 探索：让指定 Agent 与某个目标 Agent（按名字模糊匹配）开聊并评估。

    用于 `chat_with_agent` 工具：复用 _agent_chat_and_evaluate 链路，但 candidates 固定为
    名字命中的那一个，绕过 MMR / 短冷却（因为是用户明确指定的目标）。

    速率：DISCOVERY_DIRECTED_DAILY_LIMIT 控制单 Agent 每日定向调用上限。
    """
    target_query = (target_query or "").strip()
    if not target_query:
        return {"ok": False, "message": "请告诉我要找哪位 Agent（名字或关键词）"}

    me = (await session.execute(select(Agent).where(Agent.id == agent_id))).scalar_one_or_none()
    if me is None:
        return {"ok": False, "message": "Agent 不存在"}

    # 速率限制：今日已经做过几次定向
    since_today = now_utc8() - timedelta(hours=24)
    today_count = (
        await session.execute(
            select(func.count(DiscoveryAttempt.id)).where(
                and_(
                    DiscoveryAttempt.from_agent_id == agent_id,
                    DiscoveryAttempt.source == "directed",
                    DiscoveryAttempt.created_at >= since_today,
                )
            )
        )
    ).scalar() or 0
    daily_limit = max(1, settings.DISCOVERY_DIRECTED_DAILY_LIMIT)
    if today_count >= daily_limit:
        return {
            "ok": False,
            "message": f"今天主动找别人聊的次数已经用完（{daily_limit} 次/天），明天再来。",
        }

    # 按名字模糊匹配候选
    rows = (
        await session.execute(
            select(Agent).where(and_(Agent.id != agent_id, Agent.name.like(f"%{target_query}%")))
        )
    ).scalars().all()
    if not rows:
        return {"ok": False, "message": f"社区里没找到名字像 '{target_query}' 的 Agent"}
    target = rows[0]

    my_snapshot = AgentSnapshot(
        id=me.id, name=me.name, personality=strip_internal_fields(me.personality or {})
    )
    target_snapshot = AgentSnapshot(
        id=int(target.id), name=str(target.name),
        personality=strip_internal_fields(target.personality or {}),
    )

    rec = await _agent_chat_and_evaluate(
        session=session,
        agent_a=my_snapshot,
        agent_b=target_snapshot,
        llm=llm,
        min_match_score=max(0, settings.DISCOVERY_MIN_MATCH_SCORE),
        min_confidence=max(0, settings.DISCOVERY_MIN_CONFIDENCE),
    )
    await _record_attempt(
        session,
        from_agent_id=my_snapshot.id,
        to_agent_id=target_snapshot.id,
        produced_rec=bool(rec),
        score=rec["score"] if rec else None,
        source="directed",
    )
    async with write_gate(
        GatePriority.DISCOVERY,
        timeout=settings.DB_GATE_DEFAULT_TIMEOUT_SECONDS,
        label="directed_discovery_commit",
    ):
        await session.commit()

    return {
        "ok": True,
        "target_name": target_snapshot.name,
        "produced_rec": bool(rec),
        "score": rec["score"] if rec else None,
        "message": (
            f"已和 {target_snapshot.name} 聊过，对话评分 {rec['score']} 分，已生成推荐。"
            if rec
            else f"已和 {target_snapshot.name} 聊过，但本轮没达到推荐阈值。"
        ),
        "remaining_today": daily_limit - int(today_count) - 1,
    }


async def _agent_chat_and_evaluate(
    session: AsyncSession,
    agent_a: AgentSnapshot,
    agent_b: AgentSnapshot,
    llm: LLMClient,
    min_match_score: int,
    min_confidence: int,
) -> dict[str, int] | None:
    """
    Run an agent-agent conversation and evaluation.

    本函数现在使用：
    - ReAct（Yao et al. 2022）：每轮 Agent 输出 Thought / Observation / Action 结构化字段
    - Self-Consistency（Wang et al. 2022）：评估多次采样取中位数

    Important: do LLM calls first, then persist in one short DB write window.
    """
    history: list[dict[str, str]] = []
    transcript: list[tuple[int, str]] = []
    react_log: list[dict[str, str]] = []  # 仅日志：内部思考链

    starter = agent_a
    responder = agent_b

    for _ in range(4):
        starter_react = await llm.agent_chat_turn_react(starter.name, starter.personality, history)
        starter_msg = starter_react.get("action") or ""
        if not starter_msg:
            # 降级到非 ReAct 调用避免空消息
            starter_msg = await llm.agent_chat_turn(starter.name, starter.personality, history)
        history.append({"role": "user", "content": f"{starter.name}: {starter_msg}"})
        transcript.append((starter.id, starter_msg))
        react_log.append(
            {
                "speaker": starter.name,
                "thought": starter_react.get("thought", ""),
                "observation": starter_react.get("observation", ""),
                "action": starter_msg,
            }
        )

        responder_react = await llm.agent_chat_turn_react(responder.name, responder.personality, history)
        responder_msg = responder_react.get("action") or ""
        if not responder_msg:
            responder_msg = await llm.agent_chat_turn(responder.name, responder.personality, history)
        history.append({"role": "user", "content": f"{responder.name}: {responder_msg}"})
        transcript.append((responder.id, responder_msg))
        react_log.append(
            {
                "speaker": responder.name,
                "thought": responder_react.get("thought", ""),
                "observation": responder_react.get("observation", ""),
                "action": responder_msg,
            }
        )

        starter, responder = responder, starter

    evaluation = await llm.evaluate_match_self_consistent(
        agent_a.name,
        agent_a.personality,
        agent_b.name,
        agent_b.personality,
        history,
        samples=3,
    )
    compatible = bool(evaluation.get("compatible", False))
    raw_score = int(evaluation.get("score", 0) or 0)
    confidence = int(evaluation.get("confidence", 0) or 0)
    score = _calibrate_score(raw_score, confidence)
    reason = str(evaluation.get("reason", "") or "双方在对话中发现了有限共同点")

    # 第三方仲裁（JudgeAgent）：独立审查边界冲突 / 越界 / 证据不足
    judge = await llm.judge_recommendation(
        agent_a.name,
        agent_a.personality,
        agent_b.name,
        agent_b.personality,
        history,
        primary_evaluation=evaluation,
    )
    judge_pass = bool(judge.get("judge_pass", True))
    additional_risks = [str(r) for r in (judge.get("additional_risks") or []) if str(r).strip()]
    score_adjustment = int(judge.get("score_adjustment", 0) or 0)
    if score_adjustment:
        score = max(0, min(100, score + score_adjustment))

    # 把额外风险并进 evaluation.risks，便于下游展示
    merged_risks = list(evaluation.get("risks") or []) + additional_risks
    evaluation = {
        **evaluation,
        "risks": merged_risks,
        "judge": judge,
    }

    conv = Conversation(conv_type="agent_agent")
    session.add(conv)
    await session.flush()

    session.add(
        ConversationParticipant(
            conversation_id=conv.id,
            entity_type="agent",
            entity_id=agent_a.id,
        )
    )
    session.add(
        ConversationParticipant(
            conversation_id=conv.id,
            entity_type="agent",
            entity_id=agent_b.id,
        )
    )

    for sender_id, content in transcript:
        session.add(
            Message(
                conversation_id=conv.id,
                sender_role="agent",
                sender_id=sender_id,
                content=content,
            )
        )

    if compatible and judge_pass and score >= min_match_score and confidence >= min_confidence:
        rec = Recommendation(
            from_agent_id=agent_a.id,
            to_agent_id=agent_b.id,
            score=score,
            reason=reason,
            highlights=list(evaluation.get("highlights") or [])[:5],
            risks=list(evaluation.get("risks") or [])[:8],
            agent_conversation_id=conv.id,
            status="pending",
        )
        session.add(rec)
        await session.flush()
        return {
            "score": score,
            "confidence": confidence,
            "recommendation_id": rec.id,
            "react_turns": len(react_log),
            "samples": len(evaluation.get("sampled", []) or []),
            "judge_pass": judge_pass,
            "score_adjustment": score_adjustment,
        }

    await session.flush()
    return None
