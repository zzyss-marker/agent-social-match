"""头像生成的纯逻辑层。

刻意拆分两类操作，二者绝不混在一个调用里：

- request_avatar_svg(name, personality, llm, settings)：
  只做 LLM 调用，不接收任何数据库会话；超时由 AVATAR_LLM_TIMEOUT_SECONDS
  控制（旧实现是 600s，且在持有数据库事务时等待，是拖死 SQLite 的根因）。

- apply_avatar_result(session, agent, svg, error, retry_base)：
  只做数据库写回（成功头像 / 失败重试退避），不做任何耗时调用。

编排（先查资格 → 会话外调 LLM → 短事务写回）统一交给
app.services.avatar_queue.AvatarQueue。
"""

from __future__ import annotations

from datetime import timedelta
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.time_utils import ensure_utc8, now_utc8
from app.services.llm_client import LLMClient


def llm_ready(llm: LLMClient | None) -> bool:
    if llm is None:
        return False
    base = str(getattr(llm, "base_url", "") or "").strip().lower()
    return base.startswith("http://") or base.startswith("https://")


# 兼容旧名称
_llm_ready = llm_ready


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


async def request_avatar_svg(
    agent_name: str,
    personality: dict[str, Any],
    llm: LLMClient | None,
    settings: Settings,
) -> tuple[str | None, str | None]:
    """会话外调用 LLM 生成头像；返回 (svg, None) 或 (None, 错误摘要)。"""
    if not settings.AVATAR_GENERATION_ENABLED:
        return None, None
    if not llm_ready(llm):
        return None, None

    prompt = _build_avatar_prompt(agent_name, personality or {})
    timeout = max(10.0, float(settings.AVATAR_LLM_TIMEOUT_SECONDS))
    last_error: Exception | None = None
    for _ in range(2):
        try:
            raw = await llm.chat(
                prompt,
                temperature=0.2,
                max_tokens=420,
                timeout_seconds=timeout,
                connect_timeout_seconds=30.0,
            )
            return _extract_svg(raw), None
        except Exception as exc:
            last_error = exc

    return None, str(last_error or "Avatar SVG generation failed")[:240]


def avatar_retry_due(agent: Any, settings: Settings, *, now: Any | None = None) -> bool:
    """该 agent 现在是否允许尝试生成头像（不修改任何状态）。"""
    if agent.avatar_svg:
        return False
    attempts = int(agent.avatar_attempts or 0)
    if attempts >= max(1, settings.AVATAR_MAX_RETRIES):
        return False
    retry_at = ensure_utc8(agent.avatar_next_retry_at)
    if retry_at is not None and retry_at > (now or now_utc8()):
        return False
    return True


def apply_avatar_result(
    session: AsyncSession,
    agent: Any,
    *,
    svg: str | None,
    error: str | None,
    retry_base_seconds: int,
) -> None:
    """把生成结果写回 agent（只做字段赋值 + flush，由调用方负责 commit）。"""
    if svg:
        agent.avatar_svg = svg
        agent.avatar_attempts = 0
        agent.avatar_last_error = None
        agent.avatar_next_retry_at = None
    else:
        if not error:
            # 未配置 LLM / 功能关闭：不计失败，保持资格，交给下次检查
            return
        attempts = int(agent.avatar_attempts or 0) + 1
        backoff_seconds = min(
            60 * 60,
            max(5, retry_base_seconds) * (2 ** min(attempts - 1, 6)),
        )
        agent.avatar_attempts = attempts
        agent.avatar_last_error = str(error)[:240]
        agent.avatar_next_retry_at = now_utc8() + timedelta(seconds=backoff_seconds)
    session.flush()
