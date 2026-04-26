from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ── Auth ──
class UserRegister(BaseModel):
    username: str = Field(min_length=2, max_length=50)
    agent_name: str = Field(min_length=1, max_length=50)


class UserResponse(BaseModel):
    id: int
    username: str
    created_at: datetime


# ── Agent ──
class AgentResponse(BaseModel):
    id: int
    user_id: int
    name: str
    personality: dict
    status: str
    created_at: datetime


class AgentUpdate(BaseModel):
    name: str | None = None


# ── Messages ──
class MessageCreate(BaseModel):
    content: str


class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    sender_role: str
    sender_id: int
    content: str
    created_at: datetime


# ── Conversations ──
class ConversationResponse(BaseModel):
    id: int
    conv_type: str
    participants: list[dict]
    last_message: MessageResponse | None
    created_at: datetime


# ── Recommendations ──
class RecommendationResponse(BaseModel):
    id: int
    from_agent_name: str
    to_agent_name: str
    score: int
    reason: str
    from_approved: bool
    to_approved: bool
    status: str
    created_at: datetime


class DiscoveryRequest(BaseModel):
    agent_id: int


class DiscoveryResponse(BaseModel):
    new_recommendations: int
    details: list[str]
