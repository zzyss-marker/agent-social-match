"""向量语义召回服务。

把 Agent 的画像文本（traits + interests + looking_for + vibe）转成向量，
用 cosine similarity 替代纯字符串集合交集，解决"健身 ≠ 运动"这类语义近邻问题。

实现策略：
1. 真实 LLM 路径：调用 OpenAI 兼容的 /v1/embeddings 接口
2. 降级路径：当 EMBEDDING_BASE_URL 未配置或调用失败时，使用纯 Python 的字符 bigram
   bag-of-words → cosine。这种 fallback 比纯集合交集更鲁棒（"健身房" 和 "健身" 能匹配），
   且不引入外部依赖（如 sentence-transformers）。

向量缓存：写入 Agent.embedding_vector（JSON 列表）。增量更新：当 personality 文本变了
（基于 hash），重新生成。
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.models import Agent


# ---------- 公开接口 ----------

def personality_to_text(personality: dict[str, Any] | None) -> str:
    """把画像 JSON 拍成一段自然语言文本，供向量化使用。"""
    if not isinstance(personality, dict):
        return ""
    parts: list[str] = []
    for key in ("traits", "interests", "looking_for", "vibe", "context_memory"):
        val = personality.get(key)
        if isinstance(val, list):
            parts.extend(str(v) for v in val if str(v).strip())
        elif isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return " ".join(parts).strip()


def strip_internal_fields(personality: dict[str, Any] | None) -> dict[str, Any]:
    """剥离用 _ 开头的内部字段（如 _embedding_vector / _embedding_hash），
    防止泄露进 LLM prompt 和对外接口。"""
    if not isinstance(personality, dict):
        return {}
    return {k: v for k, v in personality.items() if not str(k).startswith("_")}


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


async def embed_text(
    text: str,
    settings: Settings,
    *,
    timeout_seconds: float = 30.0,
) -> list[float]:
    """优先调用配置的 embedding endpoint；失败则回退本地 bigram 向量。"""
    text_clean = (text or "").strip()
    if not text_clean:
        return []

    base_url = (getattr(settings, "EMBEDDING_BASE_URL", "") or settings.LLM_BASE_URL or "").strip()
    api_key = (getattr(settings, "EMBEDDING_API_KEY", "") or settings.LLM_API_KEY or "").strip()
    model = (getattr(settings, "EMBEDDING_MODEL", "") or "").strip()

    if base_url and model:
        try:
            return await _remote_embed(base_url, api_key, model, text_clean, timeout_seconds)
        except Exception:
            # 降级到本地
            return _local_bigram_vector(text_clean)
    return _local_bigram_vector(text_clean)


async def _remote_embed(
    base_url: str,
    api_key: str,
    model: str,
    text: str,
    timeout_seconds: float,
) -> list[float]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "input": text}
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=10.0)) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/embeddings",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    items = data.get("data") or []
    if not items:
        raise RuntimeError("embedding response missing data")
    vec = items[0].get("embedding") or []
    if not isinstance(vec, list):
        raise RuntimeError("embedding response invalid")
    return [float(v) for v in vec]


# ---------- 本地降级向量化（无外部依赖） ----------

def _local_bigram_vector(text: str, dim: int = 256) -> list[float]:
    """基于字符 bigram 的稀疏 hash 向量（投影到固定维度），再做 L2 归一化。

    - 比纯字符集合交集鲁棒：'健身' 和 '健身房' 共享 '健身' bigram
    - 完全本地，无外部依赖
    - 在 Chinese 文本上比 unigram 更具判别度
    """
    text = text.lower()
    if len(text) < 2:
        # 退化到 unigram
        bigrams = list(text)
    else:
        bigrams = [text[i : i + 2] for i in range(len(text) - 1)]

    vec = [0.0] * dim
    for bg in bigrams:
        h = int(hashlib.md5(bg.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0  # 双 hash 减少冲突
        vec[idx] += sign

    # L2 归一化
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


# ---------- 相似度 ----------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # a, b 一般已 L2 归一化；但 remote embedding 可能没归一化，安全起见再算一次
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# ---------- 缓存写入 ----------

async def ensure_agent_embedding(
    session: AsyncSession,
    agent: Agent,
    settings: Settings,
) -> list[float]:
    """如果 Agent 没向量或文本已变，重新生成并落库。返回向量。"""
    text = personality_to_text(agent.personality)
    if not text:
        return []

    new_hash = text_hash(text)
    profile = dict(agent.personality or {})
    cached_hash = str(profile.get("_embedding_hash") or "")
    cached_vec_raw = profile.get("_embedding_vector")
    if cached_hash == new_hash and isinstance(cached_vec_raw, list) and cached_vec_raw:
        return [float(v) for v in cached_vec_raw]

    vec = await embed_text(text, settings)
    profile["_embedding_hash"] = new_hash
    profile["_embedding_vector"] = vec
    agent.personality = profile
    await session.flush()
    return vec
