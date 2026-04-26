from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
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


async def run_discovery(
    session: AsyncSession,
    agent_id: int,
    llm: LLMClient,
    settings: Settings,  # kept for future tuning flags
) -> DiscoveryResponse:
    """Have an agent discover and chat with other agents."""
    lock = _DISCOVERY_LOCKS[agent_id]
    async with lock:
        result = await session.execute(select(Agent).where(Agent.id == agent_id))
        my_agent = result.scalar_one_or_none()
        if my_agent is None:
            raise ValueError("Agent not found")

        result = await session.execute(select(Agent).where(Agent.id != agent_id))
        candidates = result.scalars().all()
        if not candidates:
            return DiscoveryResponse(new_recommendations=0, details=["社区里还没有其他 Agent"])

        my_snapshot = AgentSnapshot(
            id=my_agent.id,
            name=my_agent.name,
            personality=my_agent.personality or {},
        )

        num_to_chat = min(random.randint(1, 3), len(candidates))
        selected = random.sample(candidates, num_to_chat)

        details: list[str] = []
        new_recs = 0

        for target in selected:
            target_snapshot = AgentSnapshot(
                id=target.id,
                name=target.name,
                personality=target.personality or {},
            )
            target_name = target_snapshot.name

            try:
                rec = await _agent_chat_and_evaluate(session, my_snapshot, target_snapshot, llm)
                await session.commit()
                if rec:
                    new_recs += 1
                    details.append(f"与 {target_name} 聊天后较匹配（{rec['score']}%）")
                else:
                    details.append(f"与 {target_name} 聊天后暂不推荐")
            except Exception as exc:
                await session.rollback()
                if _is_sqlite_locked_error(exc):
                    details.append(f"与 {target_name} 处理失败：数据库忙，请稍后重试")
                else:
                    details.append(f"与 {target_name} 聊天出错: {str(exc)[:80]}")

        # Write final status in a new clean transaction boundary.
        my_agent.status = "idle"
        await session.flush()

        return DiscoveryResponse(new_recommendations=new_recs, details=details)


async def _agent_chat_and_evaluate(
    session: AsyncSession,
    agent_a: AgentSnapshot,
    agent_b: AgentSnapshot,
    llm: LLMClient,
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
    score = int(evaluation.get("score", 0) or 0)
    reason = str(evaluation.get("reason", "") or "两位代理在对话中发现共同兴趣")

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

    if compatible and score >= 50:
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
        return {"score": score, "recommendation_id": rec.id}

    # Flush once so conversation transcript is persisted for debugging/audit.
    await session.flush()
    return None
