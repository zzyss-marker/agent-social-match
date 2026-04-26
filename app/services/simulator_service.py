from __future__ import annotations

import random

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.models import Agent, User
from app.schemas.schemas import AgentResponse, UserResponse
from app.services.llm_client import LLMClient

# Chinese names and personalities for simulated agents
AGENT_TEMPLATES = [
    {"name": "小美", "traits": ["温柔", "细心", "浪漫"], "interests": ["旅行", "摄影", "咖啡"], "looking_for": "一个可以分享生活点滴的伙伴", "vibe": "温暖治愈"},
    {"name": "大山", "traits": ["幽默", "开朗", "热爱运动"], "interests": ["健身", "徒步", "篮球"], "looking_for": "能一起运动的伙伴", "vibe": "阳光活力"},
    {"name": "阿乐", "traits": ["睿智", "安静", "有深度"], "interests": ["阅读", "电影", "写作"], "looking_for": "能深度交流的灵魂伴侣", "vibe": "文艺内敛"},
    {"name": "小雪", "traits": ["活泼", "好奇", "创意十足"], "interests": ["音乐", "绘画", "手工"], "looking_for": "有趣的灵魂", "vibe": "灵动创意"},
    {"name": "阿杰", "traits": ["务实", "可靠", "有担当"], "interests": ["编程", "投资", "烹饪"], "looking_for": "三观一致的伴侣", "vibe": "稳重靠谱"},
    {"name": "小云", "traits": ["自由", "冒险", "乐观"], "interests": ["旅行", "潜水", "美食"], "looking_for": "一起探索世界的搭子", "vibe": "自由奔放"},
    {"name": "木木", "traits": ["善良", "踏实", "宅"], "interests": ["动漫", "游戏", "撸猫"], "looking_for": "一起宅一起玩的伙伴", "vibe": "软萌宅系"},
    {"name": "星辰", "traits": ["理性", "独立", "有思想"], "interests": ["科技", "哲学", "天文学"], "looking_for": "能聊宇宙人生的朋友", "vibe": "理性深邃"},
]


async def generate_simulated_users(
    session: AsyncSession, count: int, settings: Settings
) -> list[dict]:
    """Generate simulated users with pre-built agent personalities."""
    count = max(1, min(count, len(AGENT_TEMPLATES)))
    templates = random.sample(AGENT_TEMPLATES, count)
    results = []

    for tmpl in templates:
        username = f"{tmpl['name']}_{random.randint(100, 999)}"
        user = User(username=username)
        session.add(user)
        await session.flush()

        agent = Agent(
            user_id=user.id,
            name=tmpl["name"],
            personality={
                "traits": tmpl["traits"],
                "interests": tmpl["interests"],
                "looking_for": tmpl["looking_for"],
                "vibe": tmpl["vibe"],
            },
            status="idle",
        )
        session.add(agent)
        await session.flush()

        results.append({
            "user": UserResponse(id=user.id, username=user.username, created_at=user.created_at),
            "agent": AgentResponse(
                id=agent.id, user_id=agent.user_id, name=agent.name,
                personality=agent.personality, status=agent.status,
                created_at=agent.created_at,
            ),
        })

    return results
