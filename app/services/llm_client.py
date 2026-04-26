from __future__ import annotations

import json
from typing import Any

import httpx

from app.core.config import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.LLM_BASE_URL.rstrip("/")
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.temperature = settings.LLM_TEMPERATURE

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=20.0)) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise RuntimeError(f"LLM request failed: HTTP {exc.response.status_code}. {detail}") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        return self._extract_message_text(response.json())

    def _extract_message_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM response missing choices.")

        choice0 = choices[0]
        if not isinstance(choice0, dict):
            raise RuntimeError("LLM response choice is invalid.")

        message = choice0.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                        text_parts.append(item["text"])
                merged = "".join(text_parts).strip()
                if merged:
                    return merged

        text = choice0.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

        raise RuntimeError("LLM response missing message content.")

    async def chat_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        text = await self.chat(messages, **kwargs)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```JSON").removeprefix("```").strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
            return {"data": parsed}
        except json.JSONDecodeError:
            return {"raw": text}

    async def extract_personality(self, conversation: list[dict[str, Any]]) -> dict[str, Any]:
        system = {
            "role": "system",
            "content": (
                "你是用户画像提炼助手。请只基于对话中的明确信息提取并仅返回 JSON："
                '{"traits":["特征1","特征2"],"interests":["兴趣1"],'
                '"looking_for":"用户在寻找什么","vibe":"相处氛围"}'
            ),
        }
        return await self.chat_json([system] + conversation, temperature=0.1, max_tokens=260)

    async def extract_user_context(self, conversation: list[dict[str, Any]]) -> dict[str, Any]:
        system = {
            "role": "system",
            "content": (
                "你是长期记忆提炼助手。只根据对话中的明确信息提取可持久化上下文，"
                "禁止猜测或编造。仅返回 JSON："
                '{"context_memory":["稳定事实1","稳定事实2"],'
                '"boundaries":["不接受项1"],'
                '"conversation_style":"用户偏好的沟通方式"}'
            ),
        }
        return await self.chat_json([system] + conversation, temperature=0.0, max_tokens=260)

    async def agent_chat_turn(
        self,
        agent_name: str,
        agent_personality: dict[str, Any],
        conversation_history: list[dict[str, Any]],
    ) -> str:
        profile = agent_personality.get("traits", [])
        interests = agent_personality.get("interests", [])
        looking_for = agent_personality.get("looking_for", "")
        vibe = agent_personality.get("vibe", "")

        system = {
            "role": "system",
            "content": (
                f"你是 {agent_name}，一个社交 AI Agent。\n"
                f"性格特征：{', '.join(profile) if profile else '暂无'}\n"
                f"兴趣爱好：{', '.join(interests) if interests else '暂无'}\n"
                f"正在寻找：{looking_for or '暂无'}\n"
                f"整体风格：{vibe or '自然友好'}\n"
                "你正在和另一位 Agent 聊天，请自然、礼貌、简洁回复，每次不超过100字。"
            ),
        }
        messages = [system] + conversation_history
        return await self.chat(messages)

    async def evaluate_match(
        self,
        agent_a_name: str,
        agent_a_profile: dict[str, Any],
        agent_b_name: str,
        agent_b_profile: dict[str, Any],
        conversation_transcript: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system = {
            "role": "system",
            "content": (
                "你是社交匹配评估助手。请根据双方资料和对话记录，返回严格、保守的评估。"
                "若证据不足必须降低分数，不得乐观打分。"
                "仅返回 JSON："
                '{"compatible":true/false,"score":0-100,"confidence":0-100,'
                '"reason":"推荐理由，100字内","highlights":["亮点1","亮点2"],'
                '"risks":["风险1","风险2"]}'
            ),
        }
        context = (
            f"{agent_a_name} 资料：{agent_a_profile}\n"
            f"{agent_b_name} 资料：{agent_b_profile}\n"
            "以下是两位 Agent 的对话记录："
        )
        messages = [system, {"role": "user", "content": context}] + conversation_transcript
        return await self.chat_json(messages, temperature=0.05, max_tokens=360)
