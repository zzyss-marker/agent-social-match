from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Agent, User
from app.schemas.schemas import AgentResponse, UserRegister, UserResponse


async def register(session: AsyncSession, data: UserRegister) -> tuple[UserResponse, AgentResponse]:
    """Create user + their personal agent."""
    user = User(username=data.username.strip())
    session.add(user)
    await session.flush()

    agent = Agent(
        user_id=user.id,
        name=data.agent_name.strip(),
        personality={
            "traits": [],
            "interests": [],
            "looking_for": "",
            "vibe": "",
            "context_memory": [],
            "boundaries": [],
            "conversation_style": "",
            "snapshots": [],
        },
        status="idle",
    )
    session.add(agent)
    await session.flush()

    return (
        UserResponse(id=user.id, username=user.username, created_at=user.created_at),
        AgentResponse(
            id=agent.id, user_id=agent.user_id, name=agent.name,
            personality=agent.personality, status=agent.status,
            created_at=agent.created_at,
        ),
    )


async def get_user(session: AsyncSession, user_id: int) -> UserResponse | None:
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        return None
    return UserResponse(id=user.id, username=user.username, created_at=user.created_at)


async def get_agent(session: AsyncSession, agent_id: int) -> AgentResponse | None:
    result = await session.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if agent is None:
        return None
    return AgentResponse(
        id=agent.id, user_id=agent.user_id, name=agent.name,
        personality=agent.personality, status=agent.status,
        created_at=agent.created_at,
    )


async def get_agent_by_user(session: AsyncSession, user_id: int) -> AgentResponse | None:
    result = await session.execute(select(Agent).where(Agent.user_id == user_id))
    agent = result.scalar_one_or_none()
    if agent is None:
        return None
    return AgentResponse(
        id=agent.id, user_id=agent.user_id, name=agent.name,
        personality=agent.personality, status=agent.status,
        created_at=agent.created_at,
    )
