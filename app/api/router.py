from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_llm, get_settings
from app.core.config import Settings
from app.models.models import Agent, ConversationParticipant
from app.schemas.schemas import (
    AgentResponse,
    ConversationResponse,
    DiscoveryRequest,
    DiscoveryResponse,
    EmailCodeRequest,
    MessageCreate,
    MessageResponse,
    RecommendationResponse,
    UserRegister,
    UserResponse,
)
from app.services import auth_service, chat_service, discovery_service, simulator_service
from app.services.llm_client import LLMClient

api_router = APIRouter(prefix="/api")


def _require_user_id(request: Request) -> int:
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return int(user_id)


# Auth
@api_router.post("/register/email-code")
async def send_register_email_code(
    data: EmailCodeRequest,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    try:
        await auth_service.send_registration_code(session, data.email, settings)
        return {"message": "验证码已发送"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail="验证码发送失败，请稍后重试")


@api_router.post("/register")
async def register(
    data: UserRegister,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    try:
        user, agent = await auth_service.register(session, data, settings)
        return {"user": user.model_dump(), "agent": agent.model_dump()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@api_router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, request: Request, session: AsyncSession = Depends(get_db)):
    current_user_id = _require_user_id(request)
    if user_id != current_user_id:
        raise HTTPException(status_code=403, detail="无权访问该用户信息")
    user = await auth_service.get_user(session, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


# Agent
@api_router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: int, request: Request, session: AsyncSession = Depends(get_db)):
    current_user_id = _require_user_id(request)
    agent = await auth_service.get_agent(session, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent 不存在")
    if agent.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="无权访问该Agent信息")
    return agent


# Conversations
@api_router.post("/conversations/user-agent")
async def start_user_agent_chat(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> ConversationResponse:
    """Get or create the user-agent conversation for current user."""
    user_id = _require_user_id(request)
    return await chat_service.get_or_create_user_agent_conv(session, user_id)


@api_router.get("/conversations")
async def list_conversations(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> list[ConversationResponse]:
    user_id = _require_user_id(request)
    return await chat_service.list_user_conversations(session, user_id)


@api_router.get("/conversations/{conv_id}/messages")
async def get_messages(
    request: Request,
    conv_id: int,
    session: AsyncSession = Depends(get_db),
) -> list[MessageResponse]:
    user_id = _require_user_id(request)
    participant = await session.execute(
        select(ConversationParticipant.id).where(
            (ConversationParticipant.conversation_id == conv_id)
            & (ConversationParticipant.entity_type == "user")
            & (ConversationParticipant.entity_id == user_id)
        )
    )
    if participant.scalar_one_or_none() is None:
        raise HTTPException(status_code=403, detail="无权访问该对话")
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
    """Send message. If user->agent, agent replies via LLM."""
    user_id = _require_user_id(request)

    conv = await chat_service.get_or_create_user_agent_conv(session, user_id)
    if conv.id != conv_id:
        raise HTTPException(status_code=403, detail="无权访问该对话")

    # Save user message
    await chat_service.send_message(session, conv_id, "user", user_id, data.content)

    # Get agent for this user
    agent = await auth_service.get_agent_by_user(session, user_id)
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent 未找到")

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
            f"回复要简短、真诚，像真人朋友。"
            f"适当提问来了解用户的喜好、性格、生活方式和感情需求。"
        ),
    }

    reply = await llm.chat_completion([system_prompt, *llm_msgs])

    # Save agent reply
    agent_msg = await chat_service.send_message(session, conv_id, "agent", agent.id, reply)

    # Incremental profile update
    user_msg_count = sum(1 for m in messages if m.sender_role == "user")
    if user_msg_count % 3 == 0:
        try:
            extracted = await llm.extract_profile_summary(
                [
                    {"role": "assistant" if m.sender_role == "agent" else "user", "content": m.content}
                    for m in messages[-18:]
                ]
            )
            result = await session.execute(select(Agent).where(Agent.id == agent.id))
            agent_model = result.scalar_one_or_none()
            if agent_model is not None:
                old = agent_model.personality if isinstance(agent_model.personality, dict) else {}
                merged = {
                    "traits": _uniq((old.get("traits") or []) + (extracted.get("traits") or []))[:10],
                    "interests": _uniq((old.get("interests") or []) + (extracted.get("interests") or []))[:10],
                    "looking_for": extracted.get("looking_for") or old.get("looking_for", ""),
                    "vibe": extracted.get("vibe") or old.get("vibe", "自然友好"),
                }
                agent_model.personality = merged
                await session.flush()
        except Exception:
            pass

    return {
        "user_message": data.content,
        "agent_message": agent_msg.model_dump(),
    }


# Discovery / Matching
@api_router.post("/discovery", response_model=DiscoveryResponse)
async def run_discovery(
    data: DiscoveryRequest,
    request: Request,
    session: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
    settings: Settings = Depends(get_settings),
):
    user_id = _require_user_id(request)

    # Ensure this agent belongs to current user
    agent = await auth_service.get_agent(session, data.agent_id)
    if agent is None or agent.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权操作该Agent")

    return await discovery_service.run_discovery(session, data.agent_id, llm, settings)


@api_router.get("/recommendations")
async def list_recommendations(
    request: Request,
    status: str | None = Query(default=None),
    session: AsyncSession = Depends(get_db),
) -> list[RecommendationResponse]:
    from app.models.models import Recommendation

    user_id = _require_user_id(request)

    # My agent
    result = await session.execute(select(Agent).where(Agent.user_id == user_id))
    my_agent = result.scalar_one_or_none()
    if my_agent is None:
        return []

    q = select(Recommendation).where(
        (Recommendation.from_agent_id == my_agent.id) | (Recommendation.to_agent_id == my_agent.id)
    )
    if status:
        q = q.where(Recommendation.status == status)

    result = await session.execute(q.order_by(Recommendation.created_at.desc()))
    recs = result.scalars().all()

    out: list[RecommendationResponse] = []
    for r in recs:
        from_agent = (await session.execute(select(Agent).where(Agent.id == r.from_agent_id))).scalar_one_or_none()
        to_agent = (await session.execute(select(Agent).where(Agent.id == r.to_agent_id))).scalar_one_or_none()
        out.append(
            RecommendationResponse(
                id=r.id,
                from_agent_name=from_agent.name if from_agent else f"Agent#{r.from_agent_id}",
                to_agent_name=to_agent.name if to_agent else f"Agent#{r.to_agent_id}",
                score=r.score,
                reason=r.reason,
                from_approved=r.from_approved,
                to_approved=r.to_approved,
                status=r.status,
                created_at=r.created_at,
            )
        )

    return out


@api_router.post("/recommendations/{rec_id}/approve")
async def approve_recommendation(
    rec_id: int,
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> dict:
    from app.models.models import Conversation, ConversationParticipant, Recommendation

    user_id = _require_user_id(request)

    # Load rec
    result = await session.execute(select(Recommendation).where(Recommendation.id == rec_id))
    rec = result.scalar_one_or_none()
    if rec is None:
        raise HTTPException(status_code=404, detail="推荐不存在")

    # My agent
    result = await session.execute(select(Agent).where(Agent.user_id == user_id))
    my_agent = result.scalar_one_or_none()
    if my_agent is None:
        raise HTTPException(status_code=400, detail="当前用户没有Agent")

    if my_agent.id not in (rec.from_agent_id, rec.to_agent_id):
        raise HTTPException(status_code=403, detail="无权操作该推荐")

    # Approve side
    if my_agent.id == rec.from_agent_id:
        rec.from_approved = True
    if my_agent.id == rec.to_agent_id:
        rec.to_approved = True

    # Mutual approved => create user_user conversation
    if rec.from_approved and rec.to_approved:
        rec.status = "mutual"

        from_agent = (await session.execute(select(Agent).where(Agent.id == rec.from_agent_id))).scalar_one()
        to_agent = (await session.execute(select(Agent).where(Agent.id == rec.to_agent_id))).scalar_one()

        conv = Conversation(conv_type="user_user")
        session.add(conv)
        await session.flush()
        session.add(ConversationParticipant(conversation_id=conv.id, entity_type="user", entity_id=from_agent.user_id))
        session.add(ConversationParticipant(conversation_id=conv.id, entity_type="user", entity_id=to_agent.user_id))

    await session.flush()

    return {
        "ok": True,
        "status": rec.status,
        "from_approved": rec.from_approved,
        "to_approved": rec.to_approved,
    }


@api_router.post("/recommendations/{rec_id}/reject")
async def reject_recommendation(
    rec_id: int,
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> dict:
    from app.models.models import Recommendation

    user_id = _require_user_id(request)

    result = await session.execute(select(Recommendation).where(Recommendation.id == rec_id))
    rec = result.scalar_one_or_none()
    if rec is None:
        raise HTTPException(status_code=404, detail="推荐不存在")

    result = await session.execute(select(Agent).where(Agent.user_id == user_id))
    my_agent = result.scalar_one_or_none()
    if my_agent is None:
        raise HTTPException(status_code=400, detail="当前用户没有Agent")

    if my_agent.id not in (rec.from_agent_id, rec.to_agent_id):
        raise HTTPException(status_code=403, detail="无权操作该推荐")

    rec.status = "rejected"
    await session.flush()

    return {"ok": True, "status": rec.status}


# Simulator
@api_router.post("/simulator/generate")
async def generate_simulated_users(
    request: Request,
    count: int = Query(default=10, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    _require_user_id(request)
    if not settings.DEBUG:
        raise HTTPException(status_code=403, detail="生产环境已禁用模拟数据生成")
    results = await simulator_service.generate_simulated_users(session, count, settings)
    return {"count": len(results), "items": results}


def _uniq(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        x = str(x).strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
