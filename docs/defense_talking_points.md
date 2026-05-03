# 答辩要点速查（Agent Social Match）

> 这份文档是给"答辩现场被老师追问时"准备的速查卡。结构：先讲项目基线，再讲 Agent 设计的 4 层能力，最后是高频问题与对应的"标准答法"。

## 一、项目一句话定位

> 这是一个面向真实社交场景的 Agent 交友匹配平台：用户先和自己的 Agent 多轮聊天沉淀画像，Agent 再后台主动和其他 Agent 对话进行匹配评估，双向同意后建立私信。

## 二、当前已落实的核心模块

| 模块 | 实现位置 | 简述 |
|---|---|---|
| Agent 身份与画像 | `app/models/models.py` `Agent.personality(JSON)` | 存 traits / interests / looking_for / vibe / context_memory / boundaries / conversation_style / snapshots |
| 用户↔Agent 多轮对话 | `app/main.py` `chat_send` + `_build_chat_system_prompt` | 截断最近 30 条历史 + system prompt → LLM；每 3 条用户消息触发画像增量抽取 |
| 画像抽取与合并 | `llm_client.py` `extract_personality / extract_user_context` + `_merge_profile` | 两个独立 prompt 各返 JSON，再去重合并保留滚动 snapshot |
| Agent↔Agent 探索 | `app/services/discovery_service.py` | 粗排 → 多轮对话 → LLM 评估 → 校准 → 阈值闸门 |
| 双向同意 → 私信 | `Recommendation` + `_handle_approval` | 双方都同意自动建 user_user 会话 |
| Tool Use（执行型 Agent） | `app/services/agent_tools.py` | Agent 可主动调用 search_similar_users / get_my_recommendations / update_my_boundary |
| ReAct + Self-Consistency | `discovery_service._agent_chat_and_evaluate` | Agent 间对话采用结构化 Thought/Action/Observation，评估多次采样取中位数 |
| 向量语义召回 | `app/services/embedding_service.py` | 画像文本向量化 + cosine 相似度，解决"健身 ≠ 运动"问题 |
| 仲裁 Agent（多 Agent 协作） | `discovery_service._run_judge_agent` | from_agent / to_agent / judge_agent 三方协作产出推荐 |
| 推荐解释卡片 | dashboard 模板 + `Recommendation.highlights/risks` | 把 evaluation 里的亮点/风险渲染成可视化卡片 |
| 安全基础 | `main.py` middleware | CSP / CSRF / 同源校验 / PBKDF2 密码哈希 / 邮箱验证码 |

## 三、Agent 的 4 层能力（被问"什么是 Agent"必答）

> **一句话定义**：Agent = LLM + 可持久化状态 + 决策策略 + 可执行动作。

1. **身份层（Identity）**：`name + personality`，定义表达风格、关注点和行为边界。
2. **记忆层（Memory）**：
   - 短期：本次对话窗口（最近 30 条消息）
   - 中期：画像 JSON（traits / interests / looking_for / vibe）
   - 长期：context_memory + snapshots 滚动快照
   - 语义层：向量化画像 + cosine 相似度检索（embedding_service）
3. **决策层（Reasoning & Policy）**：
   - LLM 评估打分 + confidence
   - `_calibrate_score` 保守校准（高分降权、低 confidence 进一步惩罚）
   - 阈值闸门（DISCOVERY_MIN_MATCH_SCORE / MIN_CONFIDENCE / COOLDOWN / MAX_PENDING）
   - Self-Consistency 多次采样取中位数
4. **执行层（Action）**：
   - 主动触发后台探索（背景任务）
   - 调用工具（Tool Use：搜索相似用户 / 查询推荐 / 更新边界）
   - 创建推荐和私信入口

## 四、Agent 间协作流程（被问"几 Agent 系统"必答）

```
Agent A (Proposer)  ─┐
                     ├──→ 4 轮 ReAct 对话 ──→ Agent A 评估 ┐
Agent B (Responder) ─┘                       Agent B 评估 ├──→ 取中位数 → JudgeAgent 仲裁 → 推荐
                                              第三次采样   ┘
```

- **A、B 是被推荐双方的 Agent**：轮流发言（结构化 Thought/Action/Observation）
- **JudgeAgent 是独立第三方**：检查边界冲突、跑题风险、对话越界，并对原始评估做修正
- **Self-Consistency**：评估调用 3 次取中位数，降低单次 LLM 输出抖动

## 五、高频追问 & 标准答法

### Q1: 你这个 Agent 和加了系统提示词的 ChatGPT 有什么区别？
**答**：ChatGPT 是无状态对话；我的 Agent 是 **LLM + 可持久化状态 + 决策策略 + 可执行动作**。具体来说：
- 状态：`personality JSON` 持久化在数据库，多轮对话累积
- 决策：不是"逢人就推"，有 `_calibrate_score` 保守校准 + 阈值闸门
- 执行：能主动触发后台探索、调用工具（search_similar_users 等）、创建推荐与私信

### Q2: 你这是几 Agent 系统？
**答**：三方协作：
- Proposer（发起方 Agent）
- Responder（响应方 Agent）
- JudgeAgent（独立仲裁，做风险检测和评估修正）
学术参考：CAMEL（角色分工对话）、AutoGen（多 Agent 编排）。

### Q3: 怎么防止 LLM 幻觉？
**答**：四道防线：
1. **system prompt 强约束**：明确"信息不足必须说不知道"，"严禁编造用户经历"
2. **保守评分**：`_calibrate_score` 把 confidence < 60 的进一步降权
3. **Self-Consistency**：评估调用 3 次取中位数，减少单次抖动
4. **JudgeAgent 仲裁**：第三方检查越界与跑题

### Q4: 推荐怎么排序？阈值怎么定？
**答**：两阶段：
1. **粗召回**：cosine 语义相似度 + 兴趣/特征集合交集（`_prefilter_score`），低成本缩池
2. **精排**：4 轮 Agent 间 ReAct 对话 + LLM 评估打分 + Self-Consistency
- 阈值（68/55）当前是工程经验值，未来可以用 valid 集做阈值搜索

### Q5: 怎么处理冷启动（用户没数据）？
**答**：靠多轮对话累积；Agent 会主动追问兴趣、关系期待、边界。每 3 条消息触发画像抽取并合并到 personality JSON。

### Q6: LLM 调用成本怎么控制？
**答**：
- max_tokens 限制 + temperature 低值（评估时用 0.05）
- 阶段化调用：粗排只用规则 + 向量，不调 LLM；精排才用
- 推荐冷却（`DISCOVERY_REC_COOLDOWN_HOURS`）防止重复探索同一对
- 限制每轮探索的对话数（`DISCOVERY_CHAT_MAX_PER_RUN`）

### Q7: 多个用户同时探索会冲突吗？
**答**：`_DISCOVERY_LOCKS` 按 agent_id 加 asyncio.Lock，避免同一 Agent 重入；每次探索结束 `agent.status = "idle"`。

### Q8: 数据库为什么 SQLite？怎么扩展？
**答**：原型阶段足够。路线图里规划升级到 PostgreSQL（生产并发更稳定）+ Redis（缓存与限流）+ Celery/RQ（任务队列）。

### Q9: 怎么防止 Agent 越界（聊政治、聊投资、聊医疗）？
**答**：
- system prompt 黑名单：明确禁止政治时事、投资理财、医疗法律建议、成人内容
- 跑题时"先简短说明只做交友画像，再把话题拉回用户本人"
- JudgeAgent 检测对话越界

### Q10: 工具调用具体怎么实现？
**答**：用 OpenAI 标准 `tools` 字段（function calling）：
- 在 LLMClient 里支持 tools 参数与 tool_calls 循环
- 工具定义放在 `agent_tools.py`：每个工具是一个 Python 函数 + JSON Schema 描述
- LLM 决定调用哪个工具 → 系统执行 → 把结果作为 `tool` role 消息回传 → LLM 继续推理
- 已注册工具：`search_similar_users(keyword)`、`get_my_recommendations()`、`update_my_boundary(item)`

### Q11: 向量召回是怎么做的？
**答**：把 Agent 的 `personality` 文本（traits + interests + looking_for + vibe 拼接）通过 `embedding_service.embed_text` 调用兼容 OpenAI 的 /v1/embeddings 接口生成向量，缓存进数据库。粗排时用 cosine 相似度替代纯字符串集合交集，能解决"健身 ≠ 运动"这类语义近邻问题。无 embedding endpoint 时回退到加权字符 n-gram 相似度。

### Q12: ReAct 是什么，你怎么用？
**答**：ReAct（Reasoning + Acting，Yao et al. 2022）让 LLM 显式输出"思考-行动-观察"三步。我把 Agent 间对话的每轮消息要求模型按 `Thought / Action / Observation` 结构产出，结合 Self-Consistency 多次采样取中位数，让推理可追溯也更稳健。

## 六、知道但暂未做的（保持诚实）

- 没有 Celery/RQ 队列（路线图）
- 没有 PostgreSQL 升级（路线图）
- 没有 SSE/WebSocket 流式输出
- 没有 A/B 测试框架
- 没有完整的安全审核管线（敏感词/PII 脱敏）

被问到"为什么没做"时统一回答："**这是原型阶段的工程取舍**：先把 Agent 的 4 层能力打透（身份/记忆/决策/执行），再做工程化扩容。路线图已列在 README。"

## 七、引用论文（被问理论支撑时用）

- **ReAct**：Yao et al., 2022. <https://arxiv.org/abs/2210.03629>
- **CAMEL**：Li et al., 2023. <https://arxiv.org/abs/2303.17760>
- **AutoGen**：Wu et al., 2023. <https://arxiv.org/abs/2308.08155>
- **Self-Consistency**：Wang et al., 2022. <https://arxiv.org/abs/2203.11171>
- **Sentence-BERT**：Reimers & Gurevych, 2019. <https://arxiv.org/abs/1908.10084>
