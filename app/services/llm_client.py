from __future__ import annotations

import json
from typing import Any

import httpx

from app.core.config import Settings


def _summarize_tool_result(name: str, result: dict[str, Any]) -> str:
    """把工具返回结果浓缩成一句给前端 chip 显示的简短描述。"""
    if not isinstance(result, dict):
        return f"{name} 执行完成"
    if "error" in result:
        return f"失败：{str(result['error'])[:60]}"

    if name == "search_similar_users":
        cnt = int(result.get("match_count", 0) or 0)
        keyword = str(result.get("keyword") or "").strip()
        list_mode = bool(result.get("list_mode"))
        if list_mode:
            return f"列出社区 Agent，共找到 {cnt} 个" if cnt else "社区里暂无其他 Agent"
        if cnt == 0:
            return f"按 '{keyword}' 搜索，无匹配"
        names = ", ".join(str(m.get("name", "")) for m in (result.get("matches") or [])[:3])
        more = "" if cnt <= 3 else " 等"
        return f"按 '{keyword}' 搜索到 {cnt} 个：{names}{more}"

    if name == "get_my_recommendations":
        cnt = int(result.get("count", 0) or 0)
        status = str(result.get("status") or "all")
        return f"查询到 {cnt} 条 {status} 推荐"

    if name == "update_my_boundary":
        if result.get("ok"):
            bds = result.get("boundaries") or []
            return f"已更新边界，共 {len(bds)} 项"
        return f"更新失败：{str(result.get('message') or '未知原因')[:60]}"

    return f"{name} 执行完成"


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
        connect_timeout = float(kwargs.pop("connect_timeout_seconds", 20.0))
        read_timeout = float(kwargs.pop("timeout_seconds", 60.0))
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(read_timeout, connect=connect_timeout)) as client:
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
            exc_name = exc.__class__.__name__
            exc_text = str(exc).strip()
            if not exc_text:
                exc_text = repr(exc)
            raise RuntimeError(f"LLM request failed: {exc_name}. {exc_text}") from exc

        return self._extract_message_text(response.json())

    async def chat_raw(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """与 chat 类似，但返回完整的 choice0.message 字典（用于 tool_calls）。

        额外支持 kwargs:
            tools: list  -> OpenAI 标准 tools 字段
            tool_choice: str|dict -> 默认 "auto"
        """
        connect_timeout = float(kwargs.pop("connect_timeout_seconds", 20.0))
        read_timeout = float(kwargs.pop("timeout_seconds", 60.0))
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)

        payload: dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(read_timeout, connect=connect_timeout)) as client:
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
            exc_name = exc.__class__.__name__
            exc_text = str(exc).strip()
            if not exc_text:
                exc_text = repr(exc)
            raise RuntimeError(f"LLM request failed: {exc_name}. {exc_text}") from exc

        data = response.json()
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise RuntimeError("LLM response missing choices.")
        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            raise RuntimeError("LLM response message invalid.")
        return message

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_dispatch: dict[str, Any],
        *,
        max_rounds: int = 3,
        **kwargs: Any,
    ) -> str:
        """兼容旧接口：仅返回最终文本。"""
        text, _ = await self.chat_with_tools_traced(
            messages, tools, tool_dispatch, max_rounds=max_rounds, **kwargs
        )
        return text

    async def chat_with_tools_traced(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_dispatch: dict[str, Any],
        *,
        max_rounds: int = 3,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]]]:
        """支持 tool_calls 循环的对话，并返回 (最终文本, 工具调用轨迹)。

        - 调用模型；如果模型返回 tool_calls，则执行对应 Python 协程，把结果以 tool role 追加，再次调用模型。
        - 最多循环 max_rounds 次，避免死循环。
        - 工具调用轨迹格式：[{"name": "...", "arguments": {...}, "summary": "...", "ok": True}]
          summary 是基于 result 提炼的简短描述，便于前端渲染。
        """
        from app.services.agent_tools import safe_parse_arguments

        history = list(messages)
        last_text = ""
        traces: list[dict[str, Any]] = []

        for _round in range(max(1, max_rounds)):
            message = await self.chat_raw(history, tools=tools, **kwargs)
            tool_calls = message.get("tool_calls")
            content = message.get("content")
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        texts.append(part["text"])
                content_text = "".join(texts).strip()
            elif isinstance(content, str):
                content_text = content.strip()
            else:
                content_text = ""

            if not tool_calls:
                return content_text, traces

            # 1) 把 assistant 的 tool_calls 消息加入 history
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content_text or None,
                "tool_calls": tool_calls,
            }
            history.append(assistant_msg)

            # 2) 执行每个 tool_call
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id") or "unknown"
                fn = call.get("function") or {}
                name = str(fn.get("name") or "")
                args = safe_parse_arguments(fn.get("arguments"))

                handler = tool_dispatch.get(name)
                if handler is None:
                    tool_result: dict[str, Any] = {"error": f"未知工具：{name}"}
                    ok = False
                else:
                    try:
                        tool_result = await handler(args)
                        ok = "error" not in tool_result
                    except Exception as exc:
                        tool_result = {"error": f"{name} 执行异常：{str(exc)[:160]}"}
                        ok = False

                traces.append(
                    {
                        "name": name,
                        "arguments": args,
                        "summary": _summarize_tool_result(name, tool_result),
                        "ok": ok,
                    }
                )

                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
            last_text = content_text

        # 超过 max_rounds 还没产出最终文本，兜底
        if last_text:
            return last_text, traces
        try:
            text = await self.chat(history, **kwargs)
            return text, traces
        except Exception:
            return last_text or "（工具调用循环结束，但未产出文本回复）", traces

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

    async def agent_chat_turn_react(
        self,
        agent_name: str,
        agent_personality: dict[str, Any],
        conversation_history: list[dict[str, Any]],
    ) -> dict[str, str]:
        """ReAct 风格的 Agent 对话回合：要求模型按 Thought / Action / Observation 结构产出。

        返回字段：
          - thought: 内部推理（仅日志，不发给对方）
          - action: 真正要发给对方的话
          - observation: 对对方上一句的解读（日志用）
        若解析失败则降级把整段文本作为 action。

        论文：ReAct (Yao et al. 2022, https://arxiv.org/abs/2210.03629)
        """
        profile = agent_personality.get("traits", [])
        interests = agent_personality.get("interests", [])
        looking_for = agent_personality.get("looking_for", "")
        vibe = agent_personality.get("vibe", "")

        system = {
            "role": "system",
            "content": (
                f"你是 {agent_name}，一个社交 AI Agent，正在和另一位 Agent 聊天评估匹配度。\n"
                f"性格特征：{', '.join(profile) if profile else '暂无'}\n"
                f"兴趣爱好：{', '.join(interests) if interests else '暂无'}\n"
                f"正在寻找：{looking_for or '暂无'}\n"
                f"整体风格：{vibe or '自然友好'}\n"
                "请严格按以下 JSON 结构输出（不要任何额外说明、不要代码块）：\n"
                '{"thought":"内部推理：你打算说什么以及为什么（25字内）",'
                '"observation":"对对方上一句的简短解读（20字内，没有就空字符串）",'
                '"action":"要发给对方的话（中文口语，自然友好，不超过80字）"}'
            ),
        }
        messages = [system] + conversation_history
        try:
            data = await self.chat_json(messages, temperature=0.4, max_tokens=260)
        except Exception:
            return {"thought": "", "observation": "", "action": ""}

        if not isinstance(data, dict) or "raw" in data:
            raw_text = ""
            if isinstance(data, dict) and isinstance(data.get("raw"), str):
                raw_text = data["raw"]
            return {"thought": "", "observation": "", "action": str(raw_text or "").strip()}

        action = str(data.get("action") or data.get("Action") or "").strip()
        thought = str(data.get("thought") or data.get("Thought") or "").strip()
        observation = str(data.get("observation") or data.get("Observation") or "").strip()
        return {"thought": thought, "observation": observation, "action": action}

    async def evaluate_match(
        self,
        agent_a_name: str,
        agent_a_profile: dict[str, Any],
        agent_b_name: str,
        agent_b_profile: dict[str, Any],
        conversation_transcript: list[dict[str, Any]],
        *,
        temperature: float = 0.05,
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
        return await self.chat_json(messages, temperature=temperature, max_tokens=360)

    async def evaluate_match_self_consistent(
        self,
        agent_a_name: str,
        agent_a_profile: dict[str, Any],
        agent_b_name: str,
        agent_b_profile: dict[str, Any],
        conversation_transcript: list[dict[str, Any]],
        *,
        samples: int = 3,
        temperatures: tuple[float, ...] = (0.0, 0.2, 0.4),
    ) -> dict[str, Any]:
        """Self-Consistency（Wang et al. 2022）：多次采样取中位数。

        - 调用 evaluate_match 多次，每次用略不同的 temperature
        - score / confidence 取中位数
        - reason / highlights / risks 取"中位 score 对应的那次"
        - 多了一个 sampled 字段记录所有采样原始结果，便于审计
        """
        import asyncio as _asyncio

        n = max(1, samples)
        temps = list(temperatures)[:n]
        while len(temps) < n:
            temps.append(0.05)

        coros = [
            self.evaluate_match(
                agent_a_name,
                agent_a_profile,
                agent_b_name,
                agent_b_profile,
                conversation_transcript,
                temperature=t,
            )
            for t in temps
        ]
        results: list[dict[str, Any]] = []
        for r in await _asyncio.gather(*coros, return_exceptions=True):
            if isinstance(r, dict):
                results.append(r)

        if not results:
            return {
                "compatible": False,
                "score": 0,
                "confidence": 0,
                "reason": "评估失败：所有采样均异常",
                "highlights": [],
                "risks": ["evaluation_failed"],
                "sampled": [],
            }

        def _safe_int(v: Any, default: int = 0) -> int:
            try:
                return max(0, min(100, int(v)))
            except Exception:
                return default

        scores = sorted(_safe_int(r.get("score", 0)) for r in results)
        confidences = sorted(_safe_int(r.get("confidence", 0)) for r in results)

        # 中位数
        mid_idx = len(scores) // 2
        median_score = scores[mid_idx] if len(scores) % 2 else (scores[mid_idx - 1] + scores[mid_idx]) // 2
        median_conf = confidences[mid_idx] if len(confidences) % 2 else (
            confidences[mid_idx - 1] + confidences[mid_idx]
        ) // 2

        # 找最接近 median_score 的一次结果，用它的 reason / highlights / risks
        anchor = min(
            results,
            key=lambda r: abs(_safe_int(r.get("score", 0)) - median_score),
        )

        compatible_votes = sum(1 for r in results if bool(r.get("compatible", False)))
        compatible = compatible_votes > len(results) // 2

        return {
            "compatible": compatible,
            "score": median_score,
            "confidence": median_conf,
            "reason": str(anchor.get("reason") or "")[:600],
            "highlights": list(anchor.get("highlights") or [])[:5],
            "risks": list(anchor.get("risks") or [])[:5],
            "sampled": [
                {
                    "score": _safe_int(r.get("score", 0)),
                    "confidence": _safe_int(r.get("confidence", 0)),
                    "compatible": bool(r.get("compatible", False)),
                }
                for r in results
            ],
        }

    async def judge_recommendation(
        self,
        agent_a_name: str,
        agent_a_profile: dict[str, Any],
        agent_b_name: str,
        agent_b_profile: dict[str, Any],
        conversation_transcript: list[dict[str, Any]],
        primary_evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        """JudgeAgent：独立第三方仲裁。

        多 Agent 协作思路对应 CAMEL（Li et al. 2023）/ AutoGen（Wu et al. 2023）：
        在原始评估之外引入一个职责单一的"风险审查 Agent"，它只看：
            - 是否有边界冲突（boundaries 中明确的不接受项）
            - 是否对话越界（涉及政治/医疗/投资/暴力等违规话题）
            - 是否存在数据脆弱（双方资料过空、对话太短等）

        返回 JSON：
            {
              "judge_pass": bool,    # 是否放行（false 表示 veto）
              "veto_reason": str,    # 若 veto 给出原因
              "additional_risks": [...],     # 补充给主评估的额外风险
              "score_adjustment": int        # [-30, 0]，最多减 30 分
            }
        """
        system = {
            "role": "system",
            "content": (
                "你是社交匹配的独立风险仲裁 Agent。你不参与对话，只做事后审查。\n"
                "你的任务：检查这条推荐是否存在以下问题，必要时投否决票（veto）。\n"
                "1. 边界冲突：任何一方 boundaries 中明确禁止的项，对方却命中了。\n"
                "2. 对话越界：转录中是否触及政治、医疗、投资、暴力、成人内容等。\n"
                "3. 证据不足：对话过短或资料过于稀疏，主评估的高分缺乏依据。\n"
                "请严格、保守、可解释。仅返回 JSON：\n"
                '{"judge_pass":true/false,"veto_reason":"<原因，不超过60字>",'
                '"additional_risks":["<额外风险1>","<额外风险2>"],'
                '"score_adjustment":<-30到0的整数，正常情况为0>}'
            ),
        }
        context = (
            f"{agent_a_name} 资料：{agent_a_profile}\n"
            f"{agent_b_name} 资料：{agent_b_profile}\n"
            f"主评估：{primary_evaluation}\n"
            "以下是双方 Agent 对话："
        )
        messages = [system, {"role": "user", "content": context}] + conversation_transcript
        try:
            data = await self.chat_json(messages, temperature=0.05, max_tokens=260)
        except Exception:
            return {
                "judge_pass": True,
                "veto_reason": "",
                "additional_risks": ["judge_agent_unavailable"],
                "score_adjustment": 0,
            }

        if not isinstance(data, dict) or "raw" in data:
            return {
                "judge_pass": True,
                "veto_reason": "",
                "additional_risks": ["judge_response_unparsed"],
                "score_adjustment": 0,
            }

        adjustment_raw = data.get("score_adjustment", 0)
        try:
            adjustment = int(adjustment_raw)
        except Exception:
            adjustment = 0
        adjustment = max(-30, min(0, adjustment))

        return {
            "judge_pass": bool(data.get("judge_pass", True)),
            "veto_reason": str(data.get("veto_reason") or "")[:200],
            "additional_risks": [str(x) for x in (data.get("additional_risks") or [])][:5],
            "score_adjustment": adjustment,
        }
