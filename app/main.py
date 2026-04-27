from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import os
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, inspect, or_, select, text
from starlette.middleware.sessions import SessionMiddleware

from app.api.router import api_router
from app.core.config import Settings, resolve_data_dir
from app.core.database import create_engine, create_session_factory
from app.core.exceptions import AppException
from app.core.logging_setup import setup_logging
from app.models.base import Base
from app.models.models import Agent, Conversation, ConversationParticipant, Message, Recommendation, User
from app.schemas.common import ErrorDetail
from app.schemas.schemas import UserRegister
from app.services import auth_service, chat_service, discovery_service, simulator_service

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

logger = structlog.get_logger(__name__)
SESSION_SECRET = os.environ.get("SESSION_SECRET", "matchmaking-dev-secret-key-2026")
_DISCOVERY_TASKS: dict[int, asyncio.Task] = {}


def LLMClient_from_settings(settings: Settings):
    from app.services.llm_client import LLMClient

    return LLMClient(settings)


def _render_home(
    templates: Jinja2Templates,
    request: Request,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"title": "首页"},
        status_code=200,
    )


def _render_login(
    templates: Jinja2Templates,
    request: Request,
    *,
    error: str | None = None,
    prefill: dict[str, str] | None = None,
) -> HTMLResponse:
    context: dict[str, Any] = {"title": "登录", "error": error}
    if prefill:
        context.update(prefill)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context=context,
        status_code=200,
    )


def _render_register(
    templates: Jinja2Templates,
    request: Request,
    *,
    error: str | None = None,
    notice: str | None = None,
    prefill: dict[str, str] | None = None,
    email_cooldown_seconds: int = 60,
) -> HTMLResponse:
    context: dict[str, Any] = {"title": "注册", "error": error, "notice": notice}
    if prefill:
        context.update(prefill)
    context["email_cooldown_seconds"] = email_cooldown_seconds
    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context=context,
        status_code=200,
    )


def _apply_startup_schema_patches(sync_conn) -> None:
    """Backfill auth-related columns/tables for existing SQLite databases."""
    inspector = inspect(sync_conn)
    table_names = set(inspector.get_table_names())

    if "users" in table_names:
        user_columns = {col["name"] for col in inspector.get_columns("users")}
        if "email" not in user_columns:
            sync_conn.execute(text("ALTER TABLE users ADD COLUMN email VARCHAR(255)"))
        if "password_hash" not in user_columns:
            sync_conn.execute(text("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255)"))
        if "email_verified" not in user_columns:
            sync_conn.execute(text("ALTER TABLE users ADD COLUMN email_verified BOOLEAN NOT NULL DEFAULT 0"))
        sync_conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_users_email ON users(email)"))

    if "email_verification_codes" not in table_names:
        sync_conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS email_verification_codes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email VARCHAR(255) NOT NULL,
                    purpose VARCHAR(32) NOT NULL,
                    code_hash VARCHAR(128) NOT NULL,
                    expires_at DATETIME NOT NULL,
                    used_at DATETIME NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )

    sync_conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_email_codes_lookup "
            "ON email_verification_codes(email, purpose, created_at)"
        )
    )


def _normalize_profile(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        profile = {}

    def _to_list(value: Any) -> list[str]:
        if isinstance(value, list):
            cleaned = [str(v).strip() for v in value if str(v).strip()]
            return cleaned[:20]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _dedupe(items: list[str], limit: int = 20) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            key = item.strip()
            if not key:
                continue
            if key not in seen:
                seen.add(key)
                out.append(key)
            if len(out) >= limit:
                break
        return out

    snapshots = profile.get("snapshots")
    if not isinstance(snapshots, list):
        snapshots = []

    context_memory = _dedupe(_to_list(profile.get("context_memory")), limit=30)
    boundaries = _dedupe(_to_list(profile.get("boundaries")), limit=20)
    conversation_style = str(profile.get("conversation_style", "")).strip()

    return {
        "traits": _to_list(profile.get("traits")),
        "interests": _to_list(profile.get("interests")),
        "looking_for": str(profile.get("looking_for", "")).strip(),
        "vibe": str(profile.get("vibe", "")).strip(),
        "context_memory": context_memory,
        "boundaries": boundaries,
        "conversation_style": conversation_style,
        "snapshots": snapshots[-20:],
    }


def _unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _merge_profile(old_profile: Any, new_profile: Any) -> dict[str, Any]:
    old_norm = _normalize_profile(old_profile)
    new_norm = _normalize_profile(new_profile)

    traits = _unique_keep_order(old_norm["traits"] + new_norm["traits"])[:20]
    interests = _unique_keep_order(old_norm["interests"] + new_norm["interests"])[:20]
    looking_for = new_norm["looking_for"] or old_norm["looking_for"]
    vibe = new_norm["vibe"] or old_norm["vibe"] or "自然友好"
    context_memory = _unique_keep_order(old_norm["context_memory"] + new_norm["context_memory"])[:30]
    boundaries = _unique_keep_order(old_norm["boundaries"] + new_norm["boundaries"])[:20]
    conversation_style = new_norm["conversation_style"] or old_norm["conversation_style"]

    snapshots = old_norm["snapshots"] + [
        {
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "traits": traits[:8],
            "interests": interests[:8],
            "looking_for": looking_for,
            "vibe": vibe,
            "context_memory": context_memory[:5],
        }
    ]

    return {
        "traits": traits,
        "interests": interests,
        "looking_for": looking_for,
        "vibe": vibe,
        "context_memory": context_memory,
        "boundaries": boundaries,
        "conversation_style": conversation_style,
        "snapshots": snapshots[-20:],
    }


def _build_chat_system_prompt(agent_name: str, personality: dict[str, Any]) -> str:
    profile = _normalize_profile(personality)
    context_memory = profile["context_memory"][:12]
    boundaries = profile["boundaries"][:8]
    return (
        f"你是 {agent_name}，用户的私人交友Agent。\n"
        f"已知用户画像：traits={profile['traits']}，interests={profile['interests']}，"
        f"looking_for={profile['looking_for']}，vibe={profile['vibe'] or '自然友好'}。\n"
        f"长期上下文记忆：{context_memory if context_memory else '暂无'}。\n"
        f"用户边界偏好：{boundaries if boundaries else '暂无'}。\n"
        f"沟通偏好：{profile['conversation_style'] or '自然、具体、少废话'}。\n"
        "核心目标：只围绕“了解用户并用于交友匹配”展开对话。\n"
        "允许话题：兴趣、生活方式、价值观、关系期待、社交边界、沟通偏好、近期状态。\n"
        "严格禁止：编造用户经历/身份信息；讨论政治时事、投资理财、医疗法律建议、成人露骨内容、"
        "暴力仇恨、与交友无关的空泛闲聊。\n"
        "若用户提无关问题：先简短说明你只做交友画像，再把话题拉回用户本人。\n"
        "若信息不足：明确说“我还不知道”，并只追问一个澄清问题。\n"
        "回复要求：中文口语、1到2句、每次不超过60字，避免说教和模板腔。"
    )


async def _run_discovery_background(app: FastAPI, agent_id: int) -> None:
    session_factory = app.state.session_factory
    try:
        async with session_factory() as session:
            try:
                await discovery_service.run_discovery(
                    session,
                    agent_id,
                    app.state.llm,
                    app.state.settings,
                )
                await session.commit()
            except Exception as exc:
                await session.rollback()
                logger.error("discovery_background_failed", agent_id=agent_id, error=str(exc), exc_info=True)

                # Best-effort status reset when discovery fails.
                agent = (
                    await session.execute(select(Agent).where(Agent.id == agent_id))
                ).scalar_one_or_none()
                if agent is not None:
                    agent.status = "idle"
                    await session.commit()
    finally:
        _DISCOVERY_TASKS.pop(agent_id, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    setup_logging(settings)
    resolve_data_dir(settings.DATABASE_URL)

    engine = create_engine(settings)
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    app.state.llm = LLMClient_from_settings(settings)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_apply_startup_schema_patches)

    logger.info("app_startup", name=settings.APP_NAME, version=settings.APP_VERSION)
    yield
    for task in list(_DISCOVERY_TASKS.values()):
        if not task.done():
            task.cancel()
    _DISCOVERY_TASKS.clear()
    await engine.dispose()
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.include_router(api_router)

    @app.exception_handler(AppException)
    async def handle_app(_: Request, exc: AppException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorDetail(code=exc.code, message=exc.message).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def handle_http(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorDetail(code="http_error", message=str(exc.detail)).model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_error", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorDetail(
                code="internal_error",
                message="服务内部错误" if not settings.DEBUG else str(exc),
            ).model_dump(),
        )

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        if request.session.get("user_id"):
            return RedirectResponse(url="/dashboard", status_code=303)
        return _render_home(templates, request)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request) -> HTMLResponse:
        if request.session.get("user_id"):
            return RedirectResponse(url="/dashboard", status_code=303)
        return _render_login(templates, request)

    @app.get("/register", response_class=HTMLResponse)
    async def register_page(request: Request) -> HTMLResponse:
        if request.session.get("user_id"):
            return RedirectResponse(url="/dashboard", status_code=303)
        return _render_register(
            templates,
            request,
            email_cooldown_seconds=request.app.state.settings.EMAIL_CODE_RESEND_COOLDOWN_SECONDS,
        )

    @app.get("/register/check-availability")
    async def register_check_availability(
        request: Request,
        field: str,
        value: str,
    ) -> JSONResponse:
        field_key = field.strip().lower()
        raw_value = value.strip()
        if field_key not in {"username", "agent_name"}:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "available": False, "message": "不支持的检查字段"},
            )

        if not raw_value:
            return JSONResponse(content={"ok": True, "available": False, "message": "请输入内容后再检查"})

        if field_key == "username" and len(raw_value) < 2:
            return JSONResponse(content={"ok": True, "available": False, "message": "用户名至少 2 个字符"})
        if len(raw_value) > 50:
            return JSONResponse(content={"ok": True, "available": False, "message": "长度不能超过 50 个字符"})

        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            if field_key == "username":
                exists_result = await session.execute(
                    select(func.count(User.id)).where(func.lower(User.username) == raw_value.lower())
                )
                available = int(exists_result.scalar_one() or 0) == 0
                message = "用户名可用" if available else "用户名已存在"
                return JSONResponse(content={"ok": True, "available": available, "message": message})

            exists_result = await session.execute(
                select(func.count(Agent.id)).where(func.lower(Agent.name) == raw_value.lower())
            )
            available = int(exists_result.scalar_one() or 0) == 0
            message = "Agent 名称可用" if available else "Agent 名称已存在"
            return JSONResponse(content={"ok": True, "available": available, "message": message})

    @app.post("/login")
    async def login(
        request: Request,
        identifier: str = Form(...),
        password: str = Form(""),
    ):
        clean_identifier = identifier.strip()
        clean_password = password.strip()
        if not clean_identifier:
            return _render_login(templates, request, error="请输入用户名或邮箱")

        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            user = await auth_service.get_user_by_identifier(session, clean_identifier)
            if user is None:
                return _render_login(
                    templates,
                    request,
                    error="账号不存在，请先注册",
                    prefill={"login_identifier": clean_identifier},
                )

            if not user.password_hash:
                return _render_login(
                    templates,
                    request,
                    error="该账号为旧版本数据，请重新注册并绑定邮箱后登录",
                    prefill={"login_identifier": clean_identifier},
                )

            if not clean_password or not auth_service.verify_password(clean_password, user.password_hash):
                return _render_login(
                    templates,
                    request,
                    error="密码错误",
                    prefill={"login_identifier": clean_identifier},
                )

            if not user.email_verified:
                return _render_login(
                    templates,
                    request,
                    error="邮箱未验证，请完成邮箱验证后再登录",
                    prefill={"login_identifier": clean_identifier},
                )

            request.session["user_id"] = user.id
        return RedirectResponse(url="/dashboard", status_code=303)

    @app.post("/register/send-code")
    async def send_register_code(
        request: Request,
        email: str = Form(...),
        username: str = Form(""),
        agent_name: str = Form(""),
    ):
        clean_email = auth_service.normalize_email(email)
        prefill = {
            "register_email": clean_email,
            "register_username": username.strip(),
            "register_agent_name": agent_name.strip(),
        }
        if not auth_service.is_valid_email(clean_email):
            return _render_register(
                templates,
                request,
                error="邮箱格式不正确",
                prefill=prefill,
                email_cooldown_seconds=request.app.state.settings.EMAIL_CODE_RESEND_COOLDOWN_SECONDS,
            )

        session_factory = request.app.state.session_factory
        settings = request.app.state.settings
        async with session_factory() as session:
            try:
                await auth_service.send_registration_code(session, clean_email, settings)
                await session.commit()
            except ValueError as exc:
                await session.rollback()
                return _render_register(
                    templates,
                    request,
                    error=str(exc),
                    prefill=prefill,
                    email_cooldown_seconds=settings.EMAIL_CODE_RESEND_COOLDOWN_SECONDS,
                )
            except Exception as exc:
                await session.rollback()
                logger.error("send_register_code_failed", error=str(exc), exc_info=True)
                return _render_register(
                    templates,
                    request,
                    error="验证码发送失败，请检查邮箱服务配置",
                    prefill=prefill,
                    email_cooldown_seconds=settings.EMAIL_CODE_RESEND_COOLDOWN_SECONDS,
                )

        return _render_register(
            templates,
            request,
            notice=f"验证码已发送至 {clean_email}，{settings.EMAIL_CODE_TTL_MINUTES} 分钟内有效",
            prefill=prefill,
            email_cooldown_seconds=settings.EMAIL_CODE_RESEND_COOLDOWN_SECONDS,
        )

    @app.post("/register")
    async def register_web(
        request: Request,
        username: str = Form(...),
        email: str = Form(...),
        password: str = Form(...),
        confirm_password: str = Form(...),
        verification_code: str = Form(...),
        agent_name: str = Form(...),
    ):
        clean_username = username.strip()
        clean_email = auth_service.normalize_email(email)
        clean_password = password.strip()
        clean_confirm_password = confirm_password.strip()
        clean_verification_code = verification_code.strip()
        clean_agent_name = agent_name.strip()
        prefill = {
            "register_username": clean_username,
            "register_email": clean_email,
            "register_agent_name": clean_agent_name,
        }

        if (
            not clean_username
            or not clean_email
            or not clean_password
            or not clean_verification_code
            or not clean_agent_name
        ):
            return _render_register(
                templates,
                request,
                error="注册信息不完整，请填写用户名、邮箱、密码、验证码和 Agent 名称",
                prefill=prefill,
                email_cooldown_seconds=request.app.state.settings.EMAIL_CODE_RESEND_COOLDOWN_SECONDS,
            )
        if clean_password != clean_confirm_password:
            return _render_register(
                templates,
                request,
                error="两次输入的密码不一致",
                prefill=prefill,
                email_cooldown_seconds=request.app.state.settings.EMAIL_CODE_RESEND_COOLDOWN_SECONDS,
            )

        session_factory = request.app.state.session_factory
        settings = request.app.state.settings
        async with session_factory() as session:
            try:
                user, _ = await auth_service.register(
                    session,
                    UserRegister(
                        username=clean_username,
                        email=clean_email,
                        password=clean_password,
                        verification_code=clean_verification_code,
                        agent_name=clean_agent_name,
                    ),
                    settings,
                )
                await session.commit()
                request.session["user_id"] = user.id
            except ValueError as exc:
                await session.rollback()
                return _render_register(
                    templates,
                    request,
                    error=str(exc),
                    prefill=prefill,
                    email_cooldown_seconds=settings.EMAIL_CODE_RESEND_COOLDOWN_SECONDS,
                )
            except Exception as exc:
                await session.rollback()
                logger.error("register_failed", error=str(exc), exc_info=True)
                return _render_register(
                    templates,
                    request,
                    error="注册失败，请稍后重试",
                    prefill=prefill,
                    email_cooldown_seconds=settings.EMAIL_CODE_RESEND_COOLDOWN_SECONDS,
                )
        return RedirectResponse(url="/dashboard", status_code=303)

    @app.get("/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse(url="/", status_code=303)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        user_id = request.session.get("user_id")
        if user_id is None:
            return RedirectResponse(url="/", status_code=303)

        session_factory = request.app.state.session_factory
        settings = request.app.state.settings
        active_tab = (request.query_params.get("tab") or "home").strip().lower()
        if active_tab not in {"home", "dm", "decision", "discovery", "community"}:
            active_tab = "home"
        community_query = (request.query_params.get("community_q") or "").strip()
        try:
            community_page = int(request.query_params.get("community_page", "1"))
        except ValueError:
            community_page = 1
        community_page = max(1, community_page)
        requested_page_size_raw = request.query_params.get("community_page_size")
        if requested_page_size_raw is not None:
            try:
                requested_page_size = int(requested_page_size_raw)
            except ValueError:
                requested_page_size = settings.COMMUNITY_PAGE_SIZE
        else:
            requested_page_size = settings.COMMUNITY_PAGE_SIZE
        community_page_size = max(1, min(requested_page_size, settings.COMMUNITY_MAX_PAGE_SIZE))
        community_offset = (community_page - 1) * community_page_size
        try:
            rec_page = int(request.query_params.get("rec_page", "1"))
        except ValueError:
            rec_page = 1
        rec_page = max(1, rec_page)
        try:
            rec_page_size_requested = int(
                request.query_params.get(
                    "rec_page_size",
                    str(settings.DASHBOARD_RECOMMENDATION_PAGE_SIZE),
                )
            )
        except ValueError:
            rec_page_size_requested = settings.DASHBOARD_RECOMMENDATION_PAGE_SIZE
        rec_page_size = max(1, min(rec_page_size_requested, settings.DASHBOARD_MAX_PANEL_PAGE_SIZE))
        rec_offset = (rec_page - 1) * rec_page_size

        try:
            disc_page = int(request.query_params.get("disc_page", "1"))
        except ValueError:
            disc_page = 1
        disc_page = max(1, disc_page)
        try:
            disc_page_size_requested = int(
                request.query_params.get(
                    "disc_page_size",
                    str(settings.DASHBOARD_DISCOVERY_PAGE_SIZE),
                )
            )
        except ValueError:
            disc_page_size_requested = settings.DASHBOARD_DISCOVERY_PAGE_SIZE
        disc_page_size = max(1, min(disc_page_size_requested, settings.DASHBOARD_MAX_PANEL_PAGE_SIZE))
        disc_offset = (disc_page - 1) * disc_page_size
        async with session_factory() as session:
            user = (await session.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
            if user is None:
                request.session.clear()
                return RedirectResponse(url="/", status_code=303)

            agent = (await session.execute(select(Agent).where(Agent.user_id == user_id))).scalar_one_or_none()
            conversations = await chat_service.list_user_conversations(session, user_id)

            recommendations: list[dict[str, Any]] = []
            recommendations_total = 0
            other_agents: list[dict[str, Any]] = []
            community_total = 0
            discovery_sessions: list[dict[str, Any]] = []
            discovery_total = 0
            direct_messages: list[dict[str, Any]] = []

            if agent:
                rec_base_where = (
                    (Recommendation.from_agent_id == agent.id)
                    | (Recommendation.to_agent_id == agent.id)
                )
                rec_count_result = await session.execute(
                    select(func.count(Recommendation.id)).where(rec_base_where)
                )
                recommendations_total = int(rec_count_result.scalar() or 0)
                rec_total_pages = max(
                    1,
                    (recommendations_total + rec_page_size - 1) // rec_page_size,
                )
                rec_page = min(rec_page, rec_total_pages)
                rec_offset = (rec_page - 1) * rec_page_size

                rec_result = await session.execute(
                    select(Recommendation)
                    .where(rec_base_where)
                    .order_by(Recommendation.created_at.desc())
                    .offset(rec_offset)
                    .limit(rec_page_size)
                )
                for rec in rec_result.scalars().all():
                    from_agent = (
                        await session.execute(select(Agent).where(Agent.id == rec.from_agent_id))
                    ).scalar_one_or_none()
                    to_agent = (
                        await session.execute(select(Agent).where(Agent.id == rec.to_agent_id))
                    ).scalar_one_or_none()
                    recommendations.append(
                        {
                            "id": rec.id,
                            "from_agent_name": from_agent.name if from_agent else "?",
                            "to_agent_name": to_agent.name if to_agent else "?",
                            "score": rec.score,
                            "reason": rec.reason,
                            "from_approved": rec.from_approved,
                            "to_approved": rec.to_approved,
                            "status": rec.status,
                            "is_from_me": agent.id == rec.from_agent_id,
                            "is_to_me": agent.id == rec.to_agent_id,
                        }
                    )

                others_query = select(Agent).where(Agent.id != agent.id)
                if community_query:
                    others_query = others_query.where(
                        or_(
                            Agent.name.ilike(f"%{community_query}%"),
                            Agent.user_id.in_(
                                select(User.id).where(User.username.ilike(f"%{community_query}%"))
                            ),
                        )
                    )
                community_total_result = await session.execute(
                    select(func.count(Agent.id)).where(others_query.whereclause)
                )
                community_total = int(community_total_result.scalar() or 0)
                community_total_pages = max(
                    1,
                    (community_total + community_page_size - 1) // community_page_size,
                )
                community_page = min(community_page, community_total_pages)
                community_offset = (community_page - 1) * community_page_size
                others_result = await session.execute(
                    others_query.order_by(Agent.created_at.desc()).offset(community_offset).limit(community_page_size)
                )
                other_agents = [
                    {
                        "id": other.id,
                        "name": other.name,
                        "personality": other.personality or {},
                        "status": other.status,
                    }
                    for other in others_result.scalars().all()
                ]

                # Discovery transcript previews (agent <-> agent conversations).
                conv_id_rows = await session.execute(
                    select(ConversationParticipant.conversation_id).where(
                        (ConversationParticipant.entity_type == "agent")
                        & (ConversationParticipant.entity_id == agent.id)
                    )
                )
                agent_conv_ids = [row[0] for row in conv_id_rows.all()]
                if agent_conv_ids:
                    discovery_total_result = await session.execute(
                        select(func.count(Conversation.id))
                        .where(
                            (Conversation.id.in_(agent_conv_ids))
                            & (Conversation.conv_type == "agent_agent")
                        )
                    )
                    discovery_total = int(discovery_total_result.scalar() or 0)
                    disc_total_pages = max(
                        1,
                        (discovery_total + disc_page_size - 1) // disc_page_size,
                    )
                    disc_page = min(disc_page, disc_total_pages)
                    disc_offset = (disc_page - 1) * disc_page_size
                    disc_conv_result = await session.execute(
                        select(Conversation)
                        .where(
                            (Conversation.id.in_(agent_conv_ids))
                            & (Conversation.conv_type == "agent_agent")
                        )
                        .order_by(Conversation.created_at.desc())
                        .offset(disc_offset)
                        .limit(disc_page_size)
                    )
                    for disc in disc_conv_result.scalars().all():
                        participant_rows = await session.execute(
                            select(ConversationParticipant).where(
                                (ConversationParticipant.conversation_id == disc.id)
                                & (ConversationParticipant.entity_type == "agent")
                            )
                        )
                        peer_agent_id = None
                        for participant in participant_rows.scalars().all():
                            if participant.entity_id != agent.id:
                                peer_agent_id = participant.entity_id
                                break

                        peer_agent = None
                        if peer_agent_id is not None:
                            peer_agent = (
                                await session.execute(select(Agent).where(Agent.id == peer_agent_id))
                            ).scalar_one_or_none()

                        msg_count = int(
                            (
                                await session.execute(
                                    select(func.count(Message.id)).where(Message.conversation_id == disc.id)
                                )
                            ).scalar()
                            or 0
                        )
                        last_message = (
                            await session.execute(
                                select(Message)
                                .where(Message.conversation_id == disc.id)
                                .order_by(Message.created_at.desc())
                                .limit(1)
                            )
                        ).scalar_one_or_none()
                        rec_for_conv = (
                            await session.execute(
                                select(Recommendation)
                                .where(Recommendation.agent_conversation_id == disc.id)
                                .order_by(Recommendation.created_at.desc())
                                .limit(1)
                            )
                        ).scalar_one_or_none()

                        discovery_sessions.append(
                            {
                                "conversation_id": disc.id,
                                "peer_agent_name": peer_agent.name if peer_agent else "Unknown",
                                "created_at": disc.created_at,
                                "message_count": msg_count,
                                "last_message": (last_message.content[:120] if last_message else ""),
                                "matched": rec_for_conv is not None,
                                "score": rec_for_conv.score if rec_for_conv else None,
                            }
                        )

                # Private user-user chats (created after mutual approval).
                user_conv_id_rows = await session.execute(
                    select(ConversationParticipant.conversation_id).where(
                        (ConversationParticipant.entity_type == "user")
                        & (ConversationParticipant.entity_id == user_id)
                    )
                )
                user_conv_ids = [row[0] for row in user_conv_id_rows.all()]
                if user_conv_ids:
                    dm_result = await session.execute(
                        select(Conversation)
                        .where(
                            (Conversation.id.in_(user_conv_ids))
                            & (Conversation.conv_type == "user_user")
                        )
                        .order_by(Conversation.created_at.desc())
                    )
                    for dm in dm_result.scalars().all():
                        dm_participants = await session.execute(
                            select(ConversationParticipant).where(
                                (ConversationParticipant.conversation_id == dm.id)
                                & (ConversationParticipant.entity_type == "user")
                            )
                        )
                        peer_user_id = None
                        for participant in dm_participants.scalars().all():
                            if participant.entity_id != user_id:
                                peer_user_id = participant.entity_id
                                break
                        if peer_user_id is None:
                            continue

                        peer_user = (
                            await session.execute(select(User).where(User.id == peer_user_id))
                        ).scalar_one_or_none()
                        peer_agent = (
                            await session.execute(select(Agent).where(Agent.user_id == peer_user_id))
                        ).scalar_one_or_none()
                        dm_last_message = (
                            await session.execute(
                                select(Message)
                                .where(Message.conversation_id == dm.id)
                                .order_by(Message.created_at.desc())
                                .limit(1)
                            )
                        ).scalar_one_or_none()

                        direct_messages.append(
                            {
                                "conversation_id": dm.id,
                                "peer_username": peer_user.username if peer_user else "Unknown user",
                                "peer_agent_name": peer_agent.name if peer_agent else "Unknown agent",
                                "last_message": dm_last_message.content[:120] if dm_last_message else "",
                                "last_message_at": dm_last_message.created_at if dm_last_message else dm.created_at,
                            }
                        )

            discovery_running = bool(
                agent
                and (
                    agent.status == "discovering"
                    or (
                        agent.id in _DISCOVERY_TASKS
                        and not _DISCOVERY_TASKS[agent.id].done()
                    )
                )
            )

        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "user": {"id": user.id, "username": user.username},
                "active_tab": active_tab,
                "agent": {
                    "id": agent.id,
                    "name": agent.name,
                    "personality": agent.personality or {},
                    "status": agent.status,
                }
                if agent
                else None,
                "conversations": [conv.model_dump() for conv in conversations],
                "recommendations": recommendations,
                "recommendations_total": recommendations_total,
                "rec_page": rec_page,
                "rec_page_size": rec_page_size,
                "rec_has_prev": rec_page > 1,
                "rec_has_next": rec_offset + len(recommendations) < recommendations_total,
                "other_agents": other_agents,
                "community_total": community_total,
                "community_query": community_query,
                "community_page": community_page,
                "community_page_size": community_page_size,
                "community_has_prev": community_page > 1,
                "community_has_next": community_offset + len(other_agents) < community_total,
                "discovery_running": discovery_running,
                "discovery_sessions": discovery_sessions,
                "discovery_total": discovery_total,
                "disc_page": disc_page,
                "disc_page_size": disc_page_size,
                "disc_has_prev": disc_page > 1,
                "disc_has_next": disc_offset + len(discovery_sessions) < discovery_total,
                "direct_messages": direct_messages,
            },
        )

    @app.get("/chat/{agent_id}", response_class=HTMLResponse)
    async def chat_page(request: Request, agent_id: int) -> HTMLResponse:
        user_id = request.session.get("user_id")
        if user_id is None:
            return RedirectResponse(url="/", status_code=303)

        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            agent = await auth_service.get_agent(session, agent_id)
            if agent is None or agent.user_id != user_id:
                raise HTTPException(status_code=404, detail="Agent不存在")

            conv = await chat_service.get_or_create_user_agent_conv(session, user_id)
            messages = await chat_service.get_messages(session, conv.id)
            await session.commit()

        return templates.TemplateResponse(
            request=request,
            name="chat.html",
            context={
                "agent": agent.model_dump(),
                "conversation_id": conv.id,
                "messages": [m.model_dump() for m in messages],
                "user_id": user_id,
            },
        )

    @app.post("/chat/{agent_id}")
    async def chat_send(
        request: Request,
        agent_id: int,
        content: str = Form(...),
    ):
        user_id = request.session.get("user_id")
        if user_id is None:
            return RedirectResponse(url="/", status_code=303)

        clean_content = (content or "").strip()
        if not clean_content:
            return RedirectResponse(url=f"/chat/{agent_id}", status_code=303)

        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            agent_resp = await auth_service.get_agent(session, agent_id)
            if agent_resp is None or agent_resp.user_id != user_id:
                raise HTTPException(status_code=404, detail="Agent不存在")

            conv = await chat_service.get_or_create_user_agent_conv(session, user_id)
            await chat_service.send_message(session, conv.id, "user", user_id, clean_content)

            messages = await chat_service.get_messages(session, conv.id)
            recent_messages = messages[-30:]
            llm_messages = [
                {"role": "assistant" if m.sender_role == "agent" else "user", "content": m.content}
                for m in recent_messages
            ]

            llm = request.app.state.llm
            system_prompt = {
                "role": "system",
                "content": _build_chat_system_prompt(agent_resp.name, agent_resp.personality or {}),
            }
            fallback_reply = "我收到了你的消息，但模型服务暂时不可用。你可以继续说，我会尽快恢复。"

            if llm is None:
                agent_reply = fallback_reply
            else:
                try:
                    agent_reply = await llm.chat(
                        [system_prompt] + llm_messages,
                        temperature=0.25,
                        max_tokens=220,
                    )
                except Exception as exc:
                    logger.warning(
                        "chat_llm_failed",
                        user_id=user_id,
                        agent_id=agent_id,
                        conversation_id=conv.id,
                        error=str(exc),
                    )
                    agent_reply = fallback_reply

            await chat_service.send_message(session, conv.id, "agent", agent_resp.id, agent_reply)

            # 每 3 条用户消息增量整理一次画像和上下文记忆。
            user_turns = sum(1 for m in recent_messages if m.sender_role == "user")
            if llm is not None and user_turns > 0 and user_turns % 3 == 0:
                try:
                    extracted_profile = await llm.extract_personality(llm_messages)
                    extracted_context = await llm.extract_user_context(llm_messages)
                    extracted: dict[str, Any] = {}
                    if isinstance(extracted_profile, dict):
                        extracted.update(extracted_profile)
                    if isinstance(extracted_context, dict):
                        extracted.update(extracted_context)
                    agent_model = (
                        await session.execute(select(Agent).where(Agent.id == agent_resp.id))
                    ).scalar_one_or_none()
                    if agent_model:
                        agent_model.personality = _merge_profile(agent_model.personality, extracted)
                except Exception as exc:
                    logger.warning(
                        "profile_extract_failed",
                        user_id=user_id,
                        agent_id=agent_id,
                        conversation_id=conv.id,
                        error=str(exc),
                    )

            await session.commit()
        return RedirectResponse(url=f"/chat/{agent_id}", status_code=303)

    @app.get("/discovery-chat/{conversation_id}", response_class=HTMLResponse)
    async def discovery_chat_page(request: Request, conversation_id: int) -> HTMLResponse:
        user_id = request.session.get("user_id")
        if user_id is None:
            return RedirectResponse(url="/", status_code=303)

        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            my_agent = (await session.execute(select(Agent).where(Agent.user_id == user_id))).scalar_one_or_none()
            if my_agent is None:
                raise HTTPException(status_code=404, detail="Agent不存在")

            conv = (
                await session.execute(
                    select(Conversation).where(
                        (Conversation.id == conversation_id)
                        & (Conversation.conv_type == "agent_agent")
                    )
                )
            ).scalar_one_or_none()
            if conv is None:
                raise HTTPException(status_code=404, detail="Conversation not found.")

            participant_rows = await session.execute(
                select(ConversationParticipant).where(
                    (ConversationParticipant.conversation_id == conversation_id)
                    & (ConversationParticipant.entity_type == "agent")
                )
            )
            participant_agent_ids = [p.entity_id for p in participant_rows.scalars().all()]
            if my_agent.id not in participant_agent_ids:
                raise HTTPException(status_code=403, detail="无权限")

            name_map: dict[int, str] = {}
            for aid in participant_agent_ids:
                agent_model = (await session.execute(select(Agent).where(Agent.id == aid))).scalar_one_or_none()
                if agent_model is not None:
                    name_map[aid] = agent_model.name

            messages = (
                await session.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at.asc())
                )
            ).scalars().all()
            rec = (
                await session.execute(
                    select(Recommendation)
                    .where(Recommendation.agent_conversation_id == conversation_id)
                    .order_by(Recommendation.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

            transcript = [
                {
                    "speaker": name_map.get(msg.sender_id, f"Agent#{msg.sender_id}"),
                    "content": msg.content,
                    "created_at": msg.created_at,
                    "is_me": msg.sender_id == my_agent.id,
                }
                for msg in messages
            ]

        return templates.TemplateResponse(
            request=request,
            name="discovery_chat.html",
            context={
                "conversation_id": conversation_id,
                "transcript": transcript,
                "matched": rec is not None,
                "score": rec.score if rec else None,
            },
        )

    @app.get("/dm/{conversation_id}", response_class=HTMLResponse)
    async def dm_page(request: Request, conversation_id: int) -> HTMLResponse:
        user_id = request.session.get("user_id")
        if user_id is None:
            return RedirectResponse(url="/", status_code=303)

        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            conv = (
                await session.execute(
                    select(Conversation).where(
                        (Conversation.id == conversation_id)
                        & (Conversation.conv_type == "user_user")
                    )
                )
            ).scalar_one_or_none()
            if conv is None:
                raise HTTPException(status_code=404, detail="Conversation not found.")

            participant_rows = await session.execute(
                select(ConversationParticipant).where(
                    (ConversationParticipant.conversation_id == conversation_id)
                    & (ConversationParticipant.entity_type == "user")
                )
            )
            participant_user_ids = [p.entity_id for p in participant_rows.scalars().all()]
            if user_id not in participant_user_ids:
                raise HTTPException(status_code=403, detail="无权限")

            peer_user_id = next((uid for uid in participant_user_ids if uid != user_id), None)
            peer_user = (
                await session.execute(select(User).where(User.id == peer_user_id))
            ).scalar_one_or_none() if peer_user_id is not None else None
            peer_agent = (
                await session.execute(select(Agent).where(Agent.user_id == peer_user_id))
            ).scalar_one_or_none() if peer_user_id is not None else None

            messages = (
                await session.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at.asc())
                )
            ).scalars().all()
            own_agent = (await session.execute(select(Agent).where(Agent.user_id == user_id))).scalar_one_or_none()

            message_rows = [
                {
                    "content": msg.content,
                    "created_at": msg.created_at,
                    "is_me": msg.sender_id == user_id,
                    "sender_name": "You" if msg.sender_id == user_id else (peer_user.username if peer_user else "Peer"),
                }
                for msg in messages
            ]

        return templates.TemplateResponse(
            request=request,
            name="dm.html",
            context={
                "conversation_id": conversation_id,
                "peer_username": peer_user.username if peer_user else "Unknown user",
                "peer_agent_name": peer_agent.name if peer_agent else "Unknown agent",
                "own_agent_name": own_agent.name if own_agent else "Agent",
                "messages": message_rows,
            },
        )

    @app.post("/dm/{conversation_id}")
    async def dm_send(
        request: Request,
        conversation_id: int,
        content: str = Form(...),
    ):
        user_id = request.session.get("user_id")
        if user_id is None:
            return RedirectResponse(url="/", status_code=303)

        clean_content = (content or "").strip()
        if not clean_content:
            return RedirectResponse(url=f"/dm/{conversation_id}", status_code=303)

        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            conv = (
                await session.execute(
                    select(Conversation).where(
                        (Conversation.id == conversation_id)
                        & (Conversation.conv_type == "user_user")
                    )
                )
            ).scalar_one_or_none()
            if conv is None:
                raise HTTPException(status_code=404, detail="Conversation not found.")

            membership = await session.execute(
                select(ConversationParticipant).where(
                    (ConversationParticipant.conversation_id == conversation_id)
                    & (ConversationParticipant.entity_type == "user")
                    & (ConversationParticipant.entity_id == user_id)
                )
            )
            if membership.scalar_one_or_none() is None:
                raise HTTPException(status_code=403, detail="无权限")

            await chat_service.send_message(session, conversation_id, "user", user_id, clean_content)
            await session.commit()
        return RedirectResponse(url=f"/dm/{conversation_id}", status_code=303)

    @app.post("/discover/{agent_id}")
    async def discover_web(request: Request, agent_id: int):
        user_id = request.session.get("user_id")
        if user_id is None:
            return RedirectResponse(url="/", status_code=303)

        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            agent = (
                await session.execute(
                    select(Agent).where((Agent.id == agent_id) & (Agent.user_id == user_id))
                )
            ).scalar_one_or_none()
            if agent is None:
                raise HTTPException(status_code=403, detail="无权限")

            active_task = _DISCOVERY_TASKS.get(agent_id)
            if active_task is not None and not active_task.done():
                return RedirectResponse(url="/dashboard", status_code=303)

            if agent.status != "discovering":
                agent.status = "discovering"
                await session.commit()
            else:
                await session.rollback()

        _DISCOVERY_TASKS[agent_id] = asyncio.create_task(
            _run_discovery_background(request.app, agent_id)
        )
        return RedirectResponse(url="/dashboard", status_code=303)

    @app.post("/rec/{rec_id}/approve")
    async def approve_web(request: Request, rec_id: int):
        return await _handle_approval(request, rec_id, approve=True)

    @app.post("/rec/{rec_id}/reject")
    async def reject_web(request: Request, rec_id: int):
        return await _handle_approval(request, rec_id, approve=False)

    @app.post("/simulate-web")
    async def simulate_web(request: Request, count: int = Form(8)):
        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            await simulator_service.generate_simulated_users(
                session,
                count,
                request.app.state.settings,
            )
            await session.commit()
        return RedirectResponse(url="/dashboard", status_code=303)

    return app


async def _handle_approval(request: Request, rec_id: int, approve: bool):
    user_id = request.session.get("user_id")
    if user_id is None:
        return RedirectResponse(url="/", status_code=303)

    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        rec = (
            await session.execute(select(Recommendation).where(Recommendation.id == rec_id))
        ).scalar_one_or_none()
        if rec is None:
            raise HTTPException(status_code=404, detail="推荐不存在")

        agent = (await session.execute(select(Agent).where(Agent.user_id == user_id))).scalar_one_or_none()
        if agent is None:
            raise HTTPException(status_code=403, detail="无权限")

        if approve:
            if agent.id == rec.from_agent_id:
                rec.from_approved = True
            elif agent.id == rec.to_agent_id:
                rec.to_approved = True
            else:
                raise HTTPException(status_code=403, detail="无权限")

            if rec.from_approved and rec.to_approved:
                rec.status = "mutual"
                from_agent = (
                    await session.execute(select(Agent).where(Agent.id == rec.from_agent_id))
                ).scalar_one()
                to_agent = (
                    await session.execute(select(Agent).where(Agent.id == rec.to_agent_id))
                ).scalar_one()

                dm = Conversation(conv_type="user_user")
                session.add(dm)
                await session.flush()
                session.add(
                    ConversationParticipant(
                        conversation_id=dm.id,
                        entity_type="user",
                        entity_id=from_agent.user_id,
                    )
                )
                session.add(
                    ConversationParticipant(
                        conversation_id=dm.id,
                        entity_type="user",
                        entity_id=to_agent.user_id,
                    )
                )
        else:
            rec.status = "rejected"

        await session.commit()
    return RedirectResponse(url="/dashboard", status_code=303)


app = create_app()
