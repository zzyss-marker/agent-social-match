from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.models import (
    Agent,
    Conversation,
    ConversationParticipant,
    Message,
    Recommendation,
)
from app.schemas.schemas import DiscoveryResponse
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


def _prefilter_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Cheap retrieval score to reduce LLM calls when agent pool is large."""
    a_traits = _to_set(a.get("traits"))
    b_traits = _to_set(b.get("traits"))
    a_interests = _to_set(a.get("interests"))
    b_interests = _to_set(b.get("interests"))
    a_looking_for = str(a.get("looking_for", "")).strip()
    b_looking_for = str(b.get("looking_for", "")).strip()
    a_vibe = str(a.get("vibe", "")).strip()
    b_vibe = str(b.get("vibe", "")).strip()

    trait_overlap = len(a_traits & b_traits)
    interest_overlap = len(a_interests & b_interests)
    looking_bonus = 1 if a_looking_for and b_looking_for and a_looking_for == b_looking_for else 0
    vibe_bonus = 1 if a_vibe and b_vibe and a_vibe == b_vibe else 0

    return (interest_overlap * 1.8) + (trait_overlap * 1.2) + (looking_bonus * 1.2) + (vibe_bonus * 0.8)


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


def _as_aware_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(UTC)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


async def _load_blocked_targets(
    session: AsyncSession,
    agent_id: int,
    cooldown_hours: int,
) -> set[int]:
    since = datetime.now(UTC) - timedelta(hours=max(1, cooldown_hours))
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

    blocked: set[int] = set()
    for from_id, to_id, status, created_at in rows:
        peer_id = to_id if from_id == agent_id else from_id
        created = _as_aware_utc(created_at)
        if status == "pending" or created >= since:
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
            personality=my_agent.personality or {},
        )

        blocked_targets = await _load_blocked_targets(
            session=session,
            agent_id=my_snapshot.id,
            cooldown_hours=settings.DISCOVERY_REC_COOLDOWN_HOURS,
        )

        scored: list[tuple[float, Agent]] = []
        for candidate in candidates:
            if candidate.id in blocked_targets:
                continue
            score = _prefilter_score(my_snapshot.personality, candidate.personality or {})
            scored.append((score, candidate))

        if not scored:
            my_agent.status = "idle"
            await session.flush()
            return DiscoveryResponse(new_recommendations=0, details=["近期已探索过主要候选，稍后再试"])

        scored.sort(key=lambda item: item[0], reverse=True)
        candidate_pool_limit = max(1, settings.DISCOVERY_CANDIDATE_POOL_LIMIT)
        candidate_pool = [item[1] for item in scored[:candidate_pool_limit]]

        min_chat = max(1, settings.DISCOVERY_CHAT_MIN_PER_RUN)
        max_chat = max(min_chat, settings.DISCOVERY_CHAT_MAX_PER_RUN)
        num_to_chat = min(random.randint(min_chat, max_chat), len(candidate_pool))
        selection_window_size = min(len(candidate_pool), max(num_to_chat * 3, 12))
        selection_window = candidate_pool[:selection_window_size]
        selected = random.sample(selection_window, num_to_chat)

        # Materialize snapshots before commits/rollbacks to avoid expired ORM lazy-load issues.
        selected_snapshots = [
            AgentSnapshot(id=int(t.id), name=str(t.name), personality=t.personality or {})
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
    Important: do LLM calls first, then persist in one short DB write window.
    """
    history: list[dict[str, str]] = []
    transcript: list[tuple[int, str]] = []

    starter = agent_a
    responder = agent_b

    for _ in range(4):
        starter_msg = await llm.agent_chat_turn(starter.name, starter.personality, history)
        history.append({"role": "user", "content": f"{starter.name}: {starter_msg}"})
        transcript.append((starter.id, starter_msg))

        responder_msg = await llm.agent_chat_turn(responder.name, responder.personality, history)
        history.append({"role": "user", "content": f"{responder.name}: {responder_msg}"})
        transcript.append((responder.id, responder_msg))

        starter, responder = responder, starter

    evaluation = await llm.evaluate_match(
        agent_a.name,
        agent_a.personality,
        agent_b.name,
        agent_b.personality,
        history,
    )
    compatible = bool(evaluation.get("compatible", False))
    raw_score = int(evaluation.get("score", 0) or 0)
    confidence = int(evaluation.get("confidence", 0) or 0)
    score = _calibrate_score(raw_score, confidence)
    reason = str(evaluation.get("reason", "") or "双方在对话中发现了有限共同点")

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

    if compatible and score >= min_match_score and confidence >= min_confidence:
        rec = Recommendation(
            from_agent_id=agent_a.id,
            to_agent_id=agent_b.id,
            score=score,
            reason=reason,
            agent_conversation_id=conv.id,
            status="pending",
        )
        session.add(rec)
        await session.flush()
        return {"score": score, "confidence": confidence, "recommendation_id": rec.id}

    await session.flush()
    return None
