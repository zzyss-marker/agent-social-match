# Agent 架构与扩展设计

> 这份文档说明项目里 Agent 的具体技术实现，以及在原始基线之上做的扩展。

## 一、Agent 抽象

```
┌─────────────────────────────────────────────────────────────┐
│                          Agent                              │
│                                                             │
│  Identity           Memory          Reasoning      Action   │
│  ─ name            ─ profile JSON  ─ LLM          ─ Tool    │
│  ─ avatar          ─ context_mem   ─ Self-Consist ─ Discover│
│  ─ personality     ─ snapshots     ─ Calibration  ─ Recommend│
│                    ─ vector(embed) ─ JudgeAgent   ─ DM      │
└─────────────────────────────────────────────────────────────┘
```

## 二、关键扩展点

### 1. Tool Use（执行型 Agent）

**位置**：`app/services/agent_tools.py` + `app/services/llm_client.py::chat_with_tools`

**实现思路**：
- 用 OpenAI 兼容的 `tools` 字段（function calling）
- 每个工具是一个 Python 函数 + JSON Schema 描述
- LLM 决定调用哪个工具 → 系统执行 → 结果回传 → LLM 继续推理
- 最多 3 轮 tool call 防止死循环

**已注册工具**：
| 工具名 | 作用 | 输入 |
|---|---|---|
| `search_similar_users` | 按关键字搜索社区里的相似 Agent | `keyword: str` |
| `get_my_recommendations` | 列出当前用户的待处理推荐 | 无 |
| `update_my_boundary` | 添加用户的交友边界偏好 | `item: str` |

**用户场景示例**：
```
用户: "帮我看看像我这种喜欢动漫的还有谁"
  ↓ LLM 决定调用 search_similar_users("动漫")
  ↓ 系统执行查询，返回 3 个候选 Agent
  ↓ LLM 整合结果回复："社区里有 木木、小雪 都喜欢动漫，要不要让我去和他们聊聊？"
```

### 2. ReAct + Self-Consistency

**位置**：`app/services/discovery_service.py::_agent_chat_and_evaluate`

**ReAct（Yao et al. 2022）**：
- 每轮 Agent 发言要求模型按 `Thought / Action / Observation` 结构产出
- Thought 是内部推理；Action 是发出去的话；Observation 是对对方上一句的解读
- 让推理过程可追溯，也让 prompt 更稳定

**Self-Consistency（Wang et al. 2022）**：
- `evaluate_match` 调用 3 次（每次 temperature 略不同）
- 对 score 和 confidence 取中位数
- 对 reason / highlights / risks 选取中位 score 对应的那次

### 3. 向量语义召回

**位置**：`app/services/embedding_service.py`

**实现**：
- 把 Agent personality 文本拼接（traits + interests + looking_for + vibe）
- 调用 `/v1/embeddings`（OpenAI 兼容）生成向量
- 缓存进 `Agent.embedding_vector` 字段（JSON 列表）
- 粗排时用 cosine similarity

**降级策略**：
- 没配置 embedding endpoint 时，回退到字符 bigram Jaccard 相似度
- 这个 fallback 比纯集合交集要鲁棒（"健身房" 和 "健身" 能匹配）

### 4. 多 Agent 仲裁（JudgeAgent）

**位置**：`app/services/discovery_service.py::_run_judge_agent`

**职责**：
- 接收 A 资料 + B 资料 + 对话转录 + 主评估结果
- 输出：风险列表、是否建议放行（veto）、调整后的 score
- 当 JudgeAgent veto 时，整条推荐被拒绝（即使主评估通过阈值）

**学术对应**：
- CAMEL（Li et al. 2023）：角色分工对话
- AutoGen（Wu et al. 2023）：多 Agent 编排

### 5. 推荐解释卡片

**位置**：`templates/dashboard.html` + `Recommendation` 模型新增字段

**实现**：
- `Recommendation` 新增 `highlights: JSON` 和 `risks: JSON` 列
- discovery 持久化时把 evaluate_match 返回的 highlights / risks 一起存
- 前端在推荐卡片上把它们渲染成可视化标签

## 三、整体调用链

```
用户在 dashboard 点 "探索"
        ↓
POST /discover/{agent_id}  (main.py)
        ↓
_run_discovery_background → discovery_service.run_discovery
        ↓
1. 加载我的 Agent + 候选池
2. 粗排：cosine similarity + 集合交集 + cooldown 过滤
3. 随机抽 N 个候选
4. 对每个候选：
   a. _agent_chat_and_evaluate
      - 4 轮 ReAct 对话（Thought/Action/Observation）
      - evaluate_match × 3 次 Self-Consistency 取中位数
   b. _run_judge_agent 仲裁（独立调用）
   c. 通过 score/confidence/judge 三重闸门 → 持久化 Recommendation（含 highlights/risks）
        ↓
用户在 dashboard 看到推荐卡片，能看到 reason + highlights + risks
        ↓
双向 approve → 自动建 user_user 会话，可进入 DM
```
