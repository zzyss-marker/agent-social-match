# Agent Social Match

> 一个面向真实社交场景的 Agent 交友匹配平台：让用户先和自己的 Agent 建立理解，再由 Agent 主动发现更匹配的人。

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0+-D71F00?logo=sqlalchemy&logoColor=white)](https://www.sqlalchemy.org/)
[![License](https://img.shields.io/badge/license-MIT-black)](#license)

## 产品定位

**Agent Social Match** 不是“聊聊天就结束”的机器人项目，而是一个完整的交友流程系统：

1. 用户先和个人 Agent 多轮对话，沉淀可持续更新的画像与偏好。
2. Agent 后台主动和其他 Agent 对话，进行发现与匹配评估。
3. 双向同意后进入私信（DM），把“推荐”变成“可行动的连接”。

## 核心价值

- **降低冷启动门槛**：用户不必先写复杂资料，先聊天再逐步完善画像。
- **减少黑盒焦虑**：可查看 Agent 发现过程中的对话转录。
- **更可控的对话边界**：模型被约束在交友画像范围，降低跑题与幻觉。
- **长期记忆可持久化**：上下文记忆存入数据库，支持持续优化推荐。

## 功能一览

| 模块 | 说明 | 当前状态 |
|---|---|---|
| 用户注册/登录 | 表单注册，自动创建专属 Agent | ✅ |
| 用户-Agent 聊天 | 多轮对话、即时回复 | ✅ |
| 画像增量抽取 | 每 3 条用户消息自动更新画像 | ✅ |
| Agent 后台发现 | 非阻塞后台任务，不卡页面 | ✅ |
| 发现过程可视化 | 支持查看 Agent-Agent 聊天转录 | ✅ |
| 双向同意机制 | 推荐双方均同意后建立关系 | ✅ |
| 私信 DM | 双向同意后自动创建用户私信入口 | ✅ |

## 系统架构

```text
Browser (Jinja2 SSR)
        |
        v
FastAPI Web/API Layer
        |
        +--> Chat Service ---------> LLMClient (OpenAI-compatible API)
        |
        +--> Discovery Service ----> Agent-Agent Conversation + Match Eval
        |
        +--> Auth/DM/Recommendation Services
        |
        v
Async SQLAlchemy (SQLite / aiosqlite)
```

## 快速开始

### 1) 准备环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2) 配置 `.env`

```bash
cp .env.example .env
```

Windows:

```powershell
Copy-Item .env.example .env
```

最少需要配置：

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `SESSION_SECRET`

### 3) 初始化数据库并启动

```bash
alembic upgrade head
uvicorn app.main:app --reload
```

访问：`http://127.0.0.1:8000`

## Docker 启动

```bash
docker compose up --build
```

## 关键路由

### Web

- `GET /` 登录/注册
- `GET /dashboard` 个人控制台
- `GET /chat/{agent_id}` 用户与 Agent 聊天
- `POST /discover/{agent_id}` 触发后台发现
- `GET /discovery-chat/{conversation_id}` 查看发现转录
- `GET /dm/{conversation_id}` 私信页面

### API (`/api`)

- `GET /api/health`
- `GET /api/health/ready`
- `POST /api/register`
- `GET /api/users/{user_id}`
- `GET /api/agents/{agent_id}`
- `POST /api/conversations/user-agent`
- `GET /api/conversations/{conv_id}/messages`
- `POST /api/conversations/{conv_id}/messages`
- `POST /api/discovery`
- `GET /api/recommendations`
- `POST /api/recommendations/{rec_id}/approve`
- `POST /api/recommendations/{rec_id}/reject`

## 对话安全与约束策略

系统提示词层面已加入以下约束：

- 只围绕用户画像与交友匹配相关话题。
- 禁止编造用户经历与身份信息。
- 对无关问题进行轻量拒答并拉回交友场景。
- 信息不足时明确“不知道”，并追问单个澄清问题。
- 回复长度与语气保持简洁自然。

> 这部分约束是产品体验的关键，不建议在未评估前移除。

## 数据与持久化

- SQLite 文件：`data/matchmaking.db`（已被 `.gitignore` 忽略）
- `Agent.personality` 持久化字段包含：
  - `traits`
  - `interests`
  - `looking_for`
  - `vibe`
  - `context_memory`
  - `boundaries`
  - `conversation_style`
  - `snapshots`

## 项目结构

```text
webtest/
├─ app/
│  ├─ main.py
│  ├─ api/
│  ├─ core/
│  ├─ models/
│  ├─ schemas/
│  └─ services/
├─ templates/
├─ static/
├─ alembic/
├─ tests/
├─ data/                # 仅保留 .gitkeep，数据库文件忽略
├─ .env.example
├─ .gitignore
└─ README.md
```

## 研发与测试

```bash
python -m compileall app
pytest -v
```

## 路线图

- [ ] 引入任务队列（Celery/RQ）替代进程内后台任务。
- [ ] 将 SQLite 升级为 PostgreSQL（生产并发更稳定）。
- [ ] 推荐解释卡片可视化（为什么推荐这个人）。
- [ ] A/B 测试不同系统提示词与推荐策略。
- [ ] 增加管理后台（审计发现任务与对话质量）。

## License

MIT
