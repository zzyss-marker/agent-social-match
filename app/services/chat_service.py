from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    Agent,
    Conversation,
    ConversationParticipant,
    Message,
)
from app.schemas.schemas import (
    ConversationResponse,
    MessageResponse,
)


async def get_or_create_user_agent_conv(
    session: AsyncSession, user_id: int
) -> ConversationResponse:
    """Find existing user-agent conversation or create one."""
    result = await session.execute(
        select(Agent).where(Agent.user_id == user_id)
    )
    agent = result.scalar_one_or_none()
    if agent is None:
        raise ValueError("Agent not found")

    # Find existing user-agent conv by querying participants
    result = await session.execute(
        select(ConversationParticipant.conversation_id)
        .where(
            (ConversationParticipant.entity_type == "user")
            & (ConversationParticipant.entity_id == user_id)
        )
    )
    conv_ids = [row[0] for row in result.all()]

    for cid in conv_ids:
        result = await session.execute(
            select(Conversation).where(
                (Conversation.id == cid)
                & (Conversation.conv_type == "user_agent")
            )
        )
        conv = result.scalar_one_or_none()
        if conv is not None:
            return await _conv_to_response(session, conv)

    # Create new conversation
    conv = Conversation(conv_type="user_agent")
    session.add(conv)
    await session.flush()

    session.add(ConversationParticipant(
        conversation_id=conv.id, entity_type="user", entity_id=user_id
    ))
    session.add(ConversationParticipant(
        conversation_id=conv.id, entity_type="agent", entity_id=agent.id
    ))
    await session.flush()

    return await _conv_to_response(session, conv)


async def send_message(
    session: AsyncSession,
    conv_id: int,
    sender_role: str,
    sender_id: int,
    content: str,
) -> MessageResponse:
    msg = Message(
        conversation_id=conv_id,
        sender_role=sender_role,
        sender_id=sender_id,
        content=content,
    )
    session.add(msg)
    await session.flush()
    return MessageResponse(
        id=msg.id, conversation_id=msg.conversation_id,
        sender_role=msg.sender_role, sender_id=msg.sender_id,
        content=msg.content, created_at=msg.created_at,
    )


async def get_messages(
    session: AsyncSession, conv_id: int, limit: int = 50
) -> list[MessageResponse]:
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conv_id)
        .order_by(Message.created_at.asc())
        .limit(limit)
    )
    return [
        MessageResponse(
            id=m.id, conversation_id=m.conversation_id,
            sender_role=m.sender_role, sender_id=m.sender_id,
            content=m.content, created_at=m.created_at,
        )
        for m in result.scalars().all()
    ]


async def list_user_conversations(
    session: AsyncSession, user_id: int
) -> list[ConversationResponse]:
    """List all conversations involving this user."""
    subq = (
        select(ConversationParticipant.conversation_id)
        .where(
            (ConversationParticipant.entity_type == "user")
            & (ConversationParticipant.entity_id == user_id)
        )
    )
    result = await session.execute(
        select(Conversation)
        .where(Conversation.id.in_(subq))
        .order_by(Conversation.created_at.desc())
    )
    user_convs = []
    for conv in result.scalars().all():
        user_convs.append(await _conv_to_response(session, conv))
    return user_convs


async def _conv_to_response(
    session: AsyncSession, conv: Conversation
) -> ConversationResponse:
    # Query participants explicitly to avoid lazy-load issues
    result = await session.execute(
        select(ConversationParticipant).where(
            ConversationParticipant.conversation_id == conv.id
        )
    )
    participant_models = result.scalars().all()
    participants = [
        {"entity_type": p.entity_type, "entity_id": p.entity_id}
        for p in participant_models
    ]

    # Last message
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conv.id)
        .order_by(Message.created_at.desc())
        .limit(1)
    )
    last_msg = result.scalar_one_or_none()
    last = None
    if last_msg:
        last = MessageResponse(
            id=last_msg.id, conversation_id=last_msg.conversation_id,
            sender_role=last_msg.sender_role, sender_id=last_msg.sender_id,
            content=last_msg.content[:100], created_at=last_msg.created_at,
        )

    return ConversationResponse(
        id=conv.id, conv_type=conv.conv_type,
        participants=participants, last_message=last,
        created_at=conv.created_at,
    )
