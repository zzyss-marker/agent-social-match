from __future__ import annotations

from datetime import timedelta
import re
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.time_utils import ensure_utc8, now_utc8
from app.models.models import Agent
from app.services.llm_client import LLMClient


def _llm_ready(llm: LLMClient | None) -> bool:
    if llm is None:
        return False
    base = str(getattr(llm, "base_url", "") or "").strip().lower()
    return base.startswith("http://") or base.startswith("https://")


def _extract_svg(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = (
            text.removeprefix("```svg")
            .removeprefix("```SVG")
            .removeprefix("```xml")
            .removeprefix("```XML")
            .removeprefix("```")
            .strip()
        )
        if text.endswith("```"):
            text = text[:-3].strip()

    # Allow optional XML declaration and case-insensitive SVG tags.
    match = re.search(r"<svg\b[\s\S]*?</svg>", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError("SVG output not found")
    svg = match.group(0).strip()

    lowered = svg.lower()
    if "<script" in lowered or "onload=" in lowered or "javascript:" in lowered:
        raise ValueError("Unsafe SVG content")
    if len(svg) > 15000:
        raise ValueError("SVG too large")

    return svg


def _build_avatar_prompt(agent_name: str, personality: dict[str, Any]) -> list[dict[str, str]]:
    traits = personality.get("traits", []) if isinstance(personality, dict) else []
    interests = personality.get("interests", []) if isinstance(personality, dict) else []
    vibe = (personality or {}).get("vibe", "") if isinstance(personality, dict) else ""
    looking_for = (personality or {}).get("looking_for", "") if isinstance(personality, dict) else ""

    system = {
        "role": "system",
        "content": (
            "You are an SVG avatar generator. Output ONLY valid SVG markup.\n"
            "No markdown, no code fences, no explanation.\n"
            "Use abstract geometric style, clean and modern, product-grade visual quality.\n"
            "Canvas must be square with viewBox='0 0 96 96'.\n"
            "Do not use external assets, scripts, or foreignObject."
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"Create a unique avatar SVG for agent '{agent_name}'.\n"
            f"Traits: {traits}\n"
            f"Interests: {interests}\n"
            f"Vibe: {vibe}\n"
            f"Looking for: {looking_for}\n"
            "Style constraints: smooth gradient background + 2-5 geometric foreground shapes + subtle highlight.\n"
            "Keep it concise and deterministic-looking."
        ),
    }
    return [system, user]


async def generate_avatar_svg_for_agent(
    session: AsyncSession,
    agent: Agent,
    llm: LLMClient,
    settings: Settings,
    *,
    force: bool = False,
) -> bool:
    if not settings.AVATAR_GENERATION_ENABLED:
        return False

    if not _llm_ready(llm):
        return False

    if agent.avatar_svg and not force:
        return True

    now = now_utc8()
    attempts = int(agent.avatar_attempts or 0)
    if not force:
        if attempts >= settings.AVATAR_MAX_RETRIES:
            return False
        retry_at = ensure_utc8(agent.avatar_next_retry_at)
        if retry_at is not None and retry_at > now:
            return False

    try:
        prompt = _build_avatar_prompt(agent.name, agent.personality or {})
        last_error: Exception | None = None
        svg: str | None = None
        for _ in range(2):
            try:
                raw = await llm.chat(
                    prompt,
                    temperature=0.2,
                    max_tokens=420,
                    timeout_seconds=600.0,
                    connect_timeout_seconds=30.0,
                )
                svg = _extract_svg(raw)
                break
            except Exception as exc:
                last_error = exc

        if not svg:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Avatar SVG generation failed")

        agent.avatar_svg = svg
        agent.avatar_attempts = 0
        agent.avatar_last_error = None
        agent.avatar_next_retry_at = None
        await session.flush()
        return True
    except Exception as exc:
        next_attempts = attempts + 1
        backoff_seconds = min(
            60 * 60,
            max(5, settings.AVATAR_RETRY_BASE_SECONDS) * (2 ** min(next_attempts - 1, 6)),
        )
        agent.avatar_attempts = next_attempts
        agent.avatar_last_error = str(exc)[:240]
        agent.avatar_next_retry_at = now + timedelta(seconds=backoff_seconds)
        await session.flush()
        return False


async def backfill_agent_avatars(
    session: AsyncSession,
    llm: LLMClient,
    settings: Settings,
) -> int:
    if not settings.AVATAR_GENERATION_ENABLED:
        return 0
    if not _llm_ready(llm):
        return 0

    now = now_utc8()
    batch_size = max(1, settings.AVATAR_BATCH_SIZE)
    max_retries = max(1, settings.AVATAR_MAX_RETRIES)

    result = await session.execute(
        select(Agent)
        .where(
            or_(Agent.avatar_svg.is_(None), Agent.avatar_svg == ""),
            Agent.avatar_attempts < max_retries,
            or_(Agent.avatar_next_retry_at.is_(None), Agent.avatar_next_retry_at <= now),
        )
        .order_by(Agent.created_at.asc())
        .limit(batch_size)
    )
    pending_agents = result.scalars().all()
    if not pending_agents:
        return 0

    generated = 0
    for agent in pending_agents:
        ok = await generate_avatar_svg_for_agent(session, agent, llm, settings, force=False)
        if ok:
            generated += 1
    return generated


async def trigger_single_agent_avatar(
    session: AsyncSession,
    agent_id: int,
    llm: LLMClient,
    settings: Settings,
) -> bool:
    result = await session.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if agent is None:
        return False
    return await generate_avatar_svg_for_agent(session, agent, llm, settings, force=True)
