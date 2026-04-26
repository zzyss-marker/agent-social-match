from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_llm, get_settings
from app.core.config import Settings
from app.schemas.schemas import (
    AgentResponse,
    ConversationResponse,
    DiscoveryRequest,
    DiscoveryResponse,
    MessageCreate,
    MessageResponse,
    RecommendationResponse,
    UserRegister,
    UserResponse,
)
from app.services import (
    auth_service,
    chat_service,
    discovery_service,
    simulator_service,
)
from app.services.llm_client import LLMClient

api_router = APIRouter(prefix="/api")


# ── Auth ──
@api_router.post("/register")
async def register(
    data: UserRegister,
    session: AsyncSession = Depends(get_db),
) -> dict:
    try:
        user, agent = await auth_service.register(session, data)
        return {"user": user.model_dump(), "agent": agent.model_dump()}
    except Exception:
        raise HTTPException(status_code=409, detail="用户名已存在")


@api_router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, session: AsyncSession = Depends(get_db)):
    user = await auth_service.get_user(session, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


# ── Agent ──
@api_router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: int, session: AsyncSession = Depends(get_db)):
    agent = await auth_service.get_agent(session, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="代理不存在")
    return agent


# ── Conversations ──
@api_router.post("/conversations/user-agent")
async def start_user_agent_chat(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> ConversationResponse:
    """Get or create the user-agent conversation for current user."""
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return await chat_service.get_or_create_user_agent_conv(session, user_id)


@api_router.get("/conversations")
async def list_conversations(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> list[ConversationResponse]:
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return await chat_service.list_user_conversations(session, user_id)


@api_router.get("/conversations/{conv_id}/messages")
async def get_messages(
    conv_id: int,
    session: AsyncSession = Depends(get_db),
) -> list[MessageResponse]:
    return await chat_service.get_messages(session, conv_id)


@api_router.post("/conversations/{conv_id}/messages")
async def send_message(
    conv_id: int,
    data: MessageCreate,
    request: Request,
    session: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Send message. If user→agent, agent replies via LLM."""
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="请先登录")

    conv = await chat_service.get_or_create_user_agent_conv(session, user_id)
    if conv.id != conv_id:
        raise HTTPException(status_code=403, detail="无权访问该对话")

    # Save user message
    await chat_service.send_message(session, conv_id, "user", user_id, data.content)

    # Get agent for this user
    agent = await auth_service.get_agent_by_user(session, user_id)
    if agent is None:
        raise HTTPException(status_code=500, detail="代理未找到")

    # Build conversation history for LLM
    messages = await chat_service.get_messages(session, conv_id)
    llm_msgs = []
    for m in messages:
        role = "assistant" if m.sender_role == "agent" else "user"
        llm_msgs.append({"role": role, "content": m.content})

    system_prompt = {
        "role": "system",
        "content": (
            f"你是 {agent.name}，用户的私人AI社交代理。"
            f"你的性格：{agent.personality.get('vibe', '友好自然')}。"
            f"你在通过对话了解用户，记住用户说的每件事。"
            f"回复要自然、温暖、简洁（100字以内），像朋友聊天一样。"
            f"适当提问来了解用户的喜好、性格、生活方式和感情需求。"
            f"当前了解的用戶档案：{agent.personality}"
        ),
    }
    full_msgs = [system_prompt] + llm_msgs

    # Get agent reply
    reply = await llm.chat(full_msgs)
    agent_msg = await chat_service.send_message(session, conv_id, "agent", agent.id, reply)

    # Periodically update agent knowledge
    if len(messages) > 0 and len(messages) % 6 == 0:
        try:
            new_personality = await llm.extract_personality(llm_msgs)
            from app.models.models import Agent as AgentModel
            from sqlalchemy import select
            result = await session.execute(select(AgentModel).where(AgentModel.id == agent.id))
            agent_model = result.scalar_one_or_none()
            if agent_model:
                agent_model.personality = new_personality
                await session.flush()
        except Exception:
            pass  # Non-critical

    return {
        "user_message": {
            "id": messages[-1].id if messages else 0,
            "conversation_id": conv_id,
            "sender_role": "user",
            "sender_id": user_id,
            "content": data.content,
        },
        "agent_reply": agent_msg.model_dump(),
    }


# ── Discovery ──
@api_router.post("/discovery", response_model=DiscoveryResponse)
async def trigger_discovery(
    data: DiscoveryRequest,
    request: Request,
    session: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
    settings: Settings = Depends(get_settings),
) -> DiscoveryResponse:
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="请先登录")

    # Verify agent belongs to this user
    agent = await auth_service.get_agent(session, data.agent_id)
    if agent is None or agent.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权操作该代理")

    return await discovery_service.run_discovery(session, data.agent_id, llm, settings)


# ── Recommendations ──
@api_router.get("/recommendations", response_model=list[RecommendationResponse])
async def list_recommendations(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> list[dict]:
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="请先登录")

    from sqlalchemy import select
    from app.models.models import Agent, Recommendation

    # Get user's agent
    result = await session.execute(select(Agent).where(Agent.user_id == user_id))
    agent = result.scalar_one_or_none()
    if agent is None:
        return []

    aid = agent.id
    result = await session.execute(
        select(Recommendation).where(
            (Recommendation.from_agent_id == aid) | (Recommendation.to_agent_id == aid)
        ).order_by(Recommendation.created_at.desc())
    )
    recs = result.scalars().all()

    output = []
    for r in recs:
        from_agent = (await session.execute(select(Agent).where(Agent.id == r.from_agent_id))).scalar_one_or_none()
        to_agent = (await session.execute(select(Agent).where(Agent.id == r.to_agent_id))).scalar_one_or_none()
        output.append(RecommendationResponse(
            id=r.id,
            from_agent_name=from_agent.name if from_agent else "?",
            to_agent_name=to_agent.name if to_agent else "?",
            score=r.score,
            reason=r.reason,
            from_approved=r.from_approved,
            to_approved=r.to_approved,
            status=r.status,
            created_at=r.created_at,
        ))
    return output


@api_router.post("/recommendations/{rec_id}/approve")
async def approve_recommendation(
    rec_id: int,
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> dict:
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="请先登录")

    from sqlalchemy import select
    from app.models.models import Agent, Conversation, ConversationParticipant, Recommendation

    result = await session.execute(select(Recommendation).where(Recommendation.id == rec_id))
    rec = result.scalar_one_or_none()
    if rec is None:
        raise HTTPException(status_code=404, detail="推荐不存在")

    # Check if this user's agent is involved
    result = await session.execute(select(Agent).where(Agent.user_id == user_id))
    agent = result.scalar_one_or_none()
    if agent is None:
        raise HTTPException(status_code=403, detail="无权限")

    if agent.id == rec.from_agent_id:
        rec.from_approved = True
    elif agent.id == rec.to_agent_id:
        rec.to_approved = True
    else:
        raise HTTPException(status_code=403, detail="无权限")

    mutual = False
    if rec.from_approved and rec.to_approved:
        rec.status = "mutual"
        mutual = True

        # Create user_user conversation for DM
        from_agent = (await session.execute(select(Agent).where(Agent.id == rec.from_agent_id))).scalar_one()
        to_agent = (await session.execute(select(Agent).where(Agent.id == rec.to_agent_id))).scalar_one()

        conv = Conversation(conv_type="user_user")
        session.add(conv)
        await session.flush()
        session.add(ConversationParticipant(conversation_id=conv.id, entity_type="user", entity_id=from_agent.user_id))
        session.add(ConversationParticipant(conversation_id=conv.id, entity_type="user", entity_id=to_agent.user_id))

    await session.flush()
    return {"status": "mutual" if mutual else "approved", "conversation_created": mutual}


@api_router.post("/recommendations/{rec_id}/reject")
async def reject_recommendation(
    rec_id: int,
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> dict:
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="请先登录")

    from sqlalchemy import select
    from app.models.models import Agent, Recommendation

    result = await session.execute(select(Recommendation).where(Recommendation.id == rec_id))
    rec = result.scalar_one_or_none()
    if rec is None:
        raise HTTPException(status_code=404, detail="推荐不存在")

    result = await session.execute(select(Agent).where(Agent.user_id == user_id))
    agent = result.scalar_one_or_none()
    if agent is None or (agent.id != rec.from_agent_id and agent.id != rec.to_agent_id):
        raise HTTPException(status_code=403, detail="无权限")

    rec.status = "rejected"
    await session.flush()
    return {"status": "rejected"}


# ── Simulation ──
@api_router.post("/simulation")
async def simulate(
    request: Request,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    count = int(request.query_params.get("count", 8))
    results = await simulator_service.generate_simulated_users(session, count, settings)
    return {"generated": len(results), "users": results}
