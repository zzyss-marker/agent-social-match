"""Agent Tool Use registry and execution.

每个工具是一个 Python 协程 + JSON Schema 描述（OpenAI tools 标准）。
LLM 决定调用哪个工具 → 系统执行 → 把结果作为 tool role 消息回传 → LLM 继续推理。
"""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Agent, Recommendation, User


# ---------- Tool implementations ----------

async def search_similar_users(
    *,
    session: AsyncSession,
    current_user_id: int,
    keyword: str,
    limit: int = 5,
) -> dict[str, Any]:
    """按关键字搜索社区里其他 Agent。
    匹配 Agent.name 或 personality 的 traits/interests/looking_for/vibe 文本。

    keyword 为空字符串时，列出社区里任意 N 个 Agent（用户问'有谁/有哪些'时）。
    """
    keyword_clean = (keyword or "").strip().lower()
    list_mode = not keyword_clean  # 空关键字 = 列表模式

    # 取 current_user 的 agent.id 排除自己
    my_agent_id = (
        await session.execute(
            select(Agent.id).where(Agent.user_id == current_user_id)
        )
    ).scalar_one_or_none()

    rows = (
        await session.execute(
            select(Agent).where(
                Agent.id != my_agent_id if my_agent_id is not None else Agent.id.is_not(None)
            ).limit(200)  # 防止全表扫描
        )
    ).scalars().all()

    cap = max(1, min(limit, 20))
    matches: list[dict[str, Any]] = []
    for ag in rows:
        haystack_parts: list[str] = [str(ag.name or "")]
        p = ag.personality or {}
        if isinstance(p, dict):
            for key in ("traits", "interests", "looking_for", "vibe"):
                value = p.get(key)
                if isinstance(value, list):
                    haystack_parts.extend(str(v) for v in value)
                elif isinstance(value, str):
                    haystack_parts.append(value)
        haystack = " ".join(haystack_parts).lower()

        if list_mode or keyword_clean in haystack:
            matches.append(
                {
                    "agent_id": ag.id,
                    "name": ag.name,
                    "interests": (p.get("interests") if isinstance(p, dict) else []) or [],
                    "vibe": (p.get("vibe") if isinstance(p, dict) else "") or "",
                }
            )
        if len(matches) >= cap:
            break

    return {
        "keyword": keyword,
        "list_mode": list_mode,
        "match_count": len(matches),
        "matches": matches,
    }


async def get_my_recommendations(
    *,
    session: AsyncSession,
    current_user_id: int,
    status: str = "pending",
) -> dict[str, Any]:
    """列出当前用户 Agent 收到/发出的推荐（默认只看 pending 的）。"""
    my_agent_id = (
        await session.execute(
            select(Agent.id).where(Agent.user_id == current_user_id)
        )
    ).scalar_one_or_none()
    if my_agent_id is None:
        return {"status": status, "items": [], "note": "当前用户未绑定 Agent"}

    base = select(Recommendation).where(
        or_(
            Recommendation.from_agent_id == my_agent_id,
            Recommendation.to_agent_id == my_agent_id,
        )
    )
    status_norm = (status or "").strip().lower()
    if status_norm in {"pending", "mutual", "rejected"}:
        base = base.where(Recommendation.status == status_norm)
    base = base.order_by(Recommendation.created_at.desc()).limit(20)

    recs = (await session.execute(base)).scalars().all()
    items: list[dict[str, Any]] = []
    for r in recs:
        peer_id = r.to_agent_id if r.from_agent_id == my_agent_id else r.from_agent_id
        peer = (
            await session.execute(select(Agent).where(Agent.id == peer_id))
        ).scalar_one_or_none()
        items.append(
            {
                "recommendation_id": r.id,
                "peer_name": peer.name if peer else f"Agent#{peer_id}",
                "score": r.score,
                "status": r.status,
                "is_outgoing": r.from_agent_id == my_agent_id,
                "from_approved": r.from_approved,
                "to_approved": r.to_approved,
            }
        )

    return {"status": status_norm or "all", "count": len(items), "items": items}


async def update_my_boundary(
    *,
    session: AsyncSession,
    current_user_id: int,
    item: str,
) -> dict[str, Any]:
    """把一项交友边界（不接受项）追加到 Agent.personality.boundaries。"""
    item_clean = (item or "").strip()
    if not item_clean:
        return {"ok": False, "message": "边界条目不能为空"}
    if len(item_clean) > 80:
        return {"ok": False, "message": "边界条目过长（最多 80 字）"}

    agent = (
        await session.execute(select(Agent).where(Agent.user_id == current_user_id))
    ).scalar_one_or_none()
    if agent is None:
        return {"ok": False, "message": "当前用户未绑定 Agent"}

    profile = dict(agent.personality or {})
    boundaries = profile.get("boundaries")
    if not isinstance(boundaries, list):
        boundaries = []
    if item_clean in boundaries:
        return {"ok": True, "message": f"边界 '{item_clean}' 已存在", "boundaries": boundaries}
    boundaries.append(item_clean)
    profile["boundaries"] = boundaries[:20]  # cap
    agent.personality = profile
    await session.flush()
    return {"ok": True, "message": f"已添加边界：{item_clean}", "boundaries": profile["boundaries"]}


async def forget_memory(
    *,
    session: AsyncSession,
    current_user_id: int,
    item: str,
    field: str = "auto",
) -> dict[str, Any]:
    """让 Agent 忘记画像中的某条具体内容（解决画像无限累积问题）。

    field 可选：
      - 'auto'（默认）：在 traits/interests/context_memory/boundaries 中查找并删除
      - 'interests' / 'traits' / 'context_memory' / 'boundaries'：仅删该字段
    """
    item_clean = (item or "").strip()
    if not item_clean:
        return {"ok": False, "message": "要忘记的内容不能为空"}

    agent = (
        await session.execute(select(Agent).where(Agent.user_id == current_user_id))
    ).scalar_one_or_none()
    if agent is None:
        return {"ok": False, "message": "当前用户未绑定 Agent"}

    profile = dict(agent.personality or {})
    candidate_fields = (
        [field]
        if field in {"traits", "interests", "context_memory", "boundaries"}
        else ["traits", "interests", "context_memory", "boundaries"]
    )
    removed: list[tuple[str, str]] = []
    for fld in candidate_fields:
        bucket = profile.get(fld)
        if not isinstance(bucket, list):
            continue
        new_bucket = []
        for v in bucket:
            text = str(v).strip()
            if text == item_clean or item_clean in text:
                removed.append((fld, text))
                continue
            new_bucket.append(v)
        profile[fld] = new_bucket

    if not removed:
        return {"ok": False, "message": f"画像里没找到包含 '{item_clean}' 的条目"}

    agent.personality = profile
    await session.flush()
    return {
        "ok": True,
        "message": f"已忘记 {len(removed)} 条画像条目",
        "removed": [{"field": f, "text": t} for f, t in removed],
    }


async def get_my_profile(
    *,
    session: AsyncSession,
    current_user_id: int,
) -> dict[str, Any]:
    """返回当前用户 Agent 的画像（剥离内部 _embedding 字段）。"""
    agent = (
        await session.execute(select(Agent).where(Agent.user_id == current_user_id))
    ).scalar_one_or_none()
    if agent is None:
        return {"ok": False, "message": "当前用户未绑定 Agent"}

    profile = agent.personality or {}
    cleaned = {k: v for k, v in profile.items() if not str(k).startswith("_")}
    return {
        "ok": True,
        "agent_name": agent.name,
        "traits": cleaned.get("traits") or [],
        "interests": cleaned.get("interests") or [],
        "looking_for": cleaned.get("looking_for") or "",
        "vibe": cleaned.get("vibe") or "",
        "boundaries": cleaned.get("boundaries") or [],
        "context_memory_count": len(cleaned.get("context_memory") or []),
    }


async def get_community_stats(
    *,
    session: AsyncSession,
    current_user_id: int,
) -> dict[str, Any]:
    """返回社区整体情况：Agent 总数、按 vibe 分布、最近新增。"""
    from sqlalchemy import func as _f
    from collections import Counter

    total_result = await session.execute(select(_f.count(Agent.id)))
    total = int(total_result.scalar_one() or 0)

    rows = (
        await session.execute(select(Agent).order_by(Agent.created_at.desc()).limit(50))
    ).scalars().all()

    vibes = Counter()
    for ag in rows:
        p = ag.personality or {}
        if isinstance(p, dict):
            v = str(p.get("vibe") or "").strip()
            if v:
                vibes[v] += 1

    recent_names = [ag.name for ag in rows[:5]]
    return {
        "ok": True,
        "total_agents": total,
        "top_vibes": [{"vibe": v, "count": c} for v, c in vibes.most_common(5)],
        "recent_5": recent_names,
    }


# ---------- OpenAI tools schema ----------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_similar_users",
            "description": (
                "搜索/列出社区里其他 Agent。"
                "用户问'有没有像我一样喜欢XX的人'/'谁喜欢XX'时，keyword 填该兴趣词；"
                "用户问'社区里都有谁'/'有哪些 Agent'/'介绍下别的'/'有谁'时，keyword 填空字符串以列出任意 Agent。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键字（兴趣/特征词），列表模式下传空字符串 ''",
                    },
                    "limit": {"type": "integer", "description": "最多返回多少条，默认 5", "default": 5},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_my_recommendations",
            "description": "列出当前用户的推荐列表。当用户问'我有什么待处理的推荐'/'有人推荐我吗'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "mutual", "rejected", "all"],
                        "description": "筛选状态，默认 pending",
                        "default": "pending",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_my_boundary",
            "description": "把一项交友边界（不接受项）添加到用户画像。当用户明确表达'我不接受X'/'我不想要XX'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "边界条目，例如 '不接受异地恋'、'不接受吸烟'"},
                },
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_memory",
            "description": (
                "让 Agent 忘记画像里的某条具体内容。"
                "当用户说'我不再喜欢XX'/'忘掉之前我说过的XX'/'删掉那条'时调用。"
                "可以删除 traits/interests/context_memory/boundaries 中的条目。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "要忘记的内容关键词"},
                    "field": {
                        "type": "string",
                        "enum": ["auto", "traits", "interests", "context_memory", "boundaries"],
                        "description": "在哪个字段查找，默认 auto 自动找",
                        "default": "auto",
                    },
                },
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_my_profile",
            "description": (
                "返回用户当前的画像（traits/interests/looking_for/vibe/boundaries 等）。"
                "当用户问'你了解我多少'/'我的画像是什么'/'你记住了我什么'时调用。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_community_stats",
            "description": (
                "返回社区整体统计：Agent 总数、最常见 vibe 分布、最近新增的 5 个 Agent。"
                "当用户问'社区多少人'/'最近活跃情况'/'有什么类型的人'时调用。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


ToolHandler = Callable[..., Awaitable[dict[str, Any]]]


def build_tool_dispatch(
    session: AsyncSession,
    current_user_id: int,
) -> dict[str, ToolHandler]:
    """生成工具名 -> 协程 的派发表，已绑定 session 与当前用户。"""

    async def _search(arguments: dict[str, Any]) -> dict[str, Any]:
        return await search_similar_users(
            session=session,
            current_user_id=current_user_id,
            keyword=str(arguments.get("keyword", "")),
            limit=int(arguments.get("limit", 5) or 5),
        )

    async def _get_recs(arguments: dict[str, Any]) -> dict[str, Any]:
        return await get_my_recommendations(
            session=session,
            current_user_id=current_user_id,
            status=str(arguments.get("status", "pending")),
        )

    async def _update_boundary(arguments: dict[str, Any]) -> dict[str, Any]:
        return await update_my_boundary(
            session=session,
            current_user_id=current_user_id,
            item=str(arguments.get("item", "")),
        )

    async def _forget(arguments: dict[str, Any]) -> dict[str, Any]:
        return await forget_memory(
            session=session,
            current_user_id=current_user_id,
            item=str(arguments.get("item", "")),
            field=str(arguments.get("field", "auto")),
        )

    async def _profile(arguments: dict[str, Any]) -> dict[str, Any]:
        return await get_my_profile(session=session, current_user_id=current_user_id)

    async def _stats(arguments: dict[str, Any]) -> dict[str, Any]:
        return await get_community_stats(session=session, current_user_id=current_user_id)

    return {
        "search_similar_users": _search,
        "get_my_recommendations": _get_recs,
        "update_my_boundary": _update_boundary,
        "forget_memory": _forget,
        "get_my_profile": _profile,
        "get_community_stats": _stats,
    }


def safe_parse_arguments(raw: Any) -> dict[str, Any]:
    """OpenAI tool arguments 通常是 JSON 字符串，这里安全解析。"""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}
