from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import re
import secrets
import smtplib
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage

from sqlalchemy import desc, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.models import Agent, EmailVerificationCode, User
from app.schemas.schemas import AgentResponse, UserRegister, UserResponse

_EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,63}$")
_REGISTER_PURPOSE = "register"
_PBKDF2_ITERATIONS = 240_000


def normalize_email(email: str) -> str:
    return email.strip().lower()


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_PATTERN.fullmatch(normalize_email(email)))


def validate_password(password: str, min_length: int) -> tuple[bool, str]:
    if len(password) < min_length:
        return False, f"密码长度至少 {min_length} 位"
    if not any(ch.isalpha() for ch in password):
        return False, "密码至少包含一个字母"
    if not any(ch.isdigit() for ch in password):
        return False, "密码至少包含一个数字"
    return True, ""


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    salt_b64 = base64.b64encode(salt).decode("ascii")
    digest_b64 = base64.b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt_b64}${digest_b64}"


def verify_password(password: str, encoded_hash: str | None) -> bool:
    if not encoded_hash:
        return False
    try:
        algorithm, iterations_raw, salt_b64, digest_b64 = encoded_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
    except Exception:
        return False

    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def _hash_email_code(email: str, code: str, purpose: str, secret: str) -> str:
    payload = f"{normalize_email(email)}:{purpose}:{code}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def _generate_code() -> str:
    return f"{secrets.randbelow(900000) + 100000}"


def _resolve_mail_sender(settings: Settings) -> str:
    sender = settings.SMTP_SENDER.strip()
    if sender:
        return sender
    user = settings.SMTP_USER.strip()
    if user:
        return user
    return ""


def _send_email_code_sync(settings: Settings, to_email: str, code: str) -> None:
    host = settings.SMTP_HOST.strip()
    sender = _resolve_mail_sender(settings)
    if not host or not sender:
        raise RuntimeError("邮箱服务未配置，请在 .env 中设置 SMTP_HOST / SMTP_USER / SMTP_SENDER")

    message = EmailMessage()
    message["Subject"] = f"{settings.APP_NAME} 注册验证码"
    message["From"] = sender
    message["To"] = to_email
    message.set_content(
        "你正在进行 Agent Social Match 注册。\n\n"
        f"验证码：{code}\n"
        f"有效期：{settings.EMAIL_CODE_TTL_MINUTES} 分钟\n\n"
        "如果不是你本人操作，请忽略此邮件。"
    )

    password = settings.SMTP_PASSWORD
    username = settings.SMTP_USER.strip()

    if settings.SMTP_USE_SSL:
        with smtplib.SMTP_SSL(host, settings.SMTP_PORT, timeout=12) as server:
            if username:
                server.login(username, password)
            server.send_message(message)
        return

    with smtplib.SMTP(host, settings.SMTP_PORT, timeout=12) as server:
        if settings.SMTP_USE_STARTTLS:
            server.starttls()
        if username:
            server.login(username, password)
        server.send_message(message)


async def send_registration_code(session: AsyncSession, email: str, settings: Settings) -> None:
    clean_email = normalize_email(email)
    if not is_valid_email(clean_email):
        raise ValueError("邮箱格式不正确")

    existing_user = await session.execute(select(User.id).where(User.email == clean_email))
    if existing_user.scalar_one_or_none() is not None:
        raise ValueError("该邮箱已注册")

    now = datetime.now(UTC)
    cooldown_since = now - timedelta(seconds=settings.EMAIL_CODE_RESEND_COOLDOWN_SECONDS)
    hourly_since = now - timedelta(hours=1)
    daily_since = now - timedelta(days=1)

    recent_code = await session.execute(
        select(EmailVerificationCode.id)
        .where(
            EmailVerificationCode.email == clean_email,
            EmailVerificationCode.purpose == _REGISTER_PURPOSE,
            EmailVerificationCode.created_at >= cooldown_since,
        )
        .order_by(desc(EmailVerificationCode.created_at))
        .limit(1)
    )
    if recent_code.scalar_one_or_none() is not None:
        raise ValueError(
            f"验证码发送过于频繁，请 {settings.EMAIL_CODE_RESEND_COOLDOWN_SECONDS} 秒后重试"
        )

    hourly_count_result = await session.execute(
        select(func.count(EmailVerificationCode.id)).where(
            EmailVerificationCode.email == clean_email,
            EmailVerificationCode.purpose == _REGISTER_PURPOSE,
            EmailVerificationCode.created_at >= hourly_since,
        )
    )
    hourly_count = int(hourly_count_result.scalar_one() or 0)
    if hourly_count >= settings.EMAIL_CODE_HOURLY_LIMIT:
        raise ValueError(
            f"该邮箱 1 小时内最多发送 {settings.EMAIL_CODE_HOURLY_LIMIT} 次验证码，请稍后再试"
        )

    daily_count_result = await session.execute(
        select(func.count(EmailVerificationCode.id)).where(
            EmailVerificationCode.email == clean_email,
            EmailVerificationCode.purpose == _REGISTER_PURPOSE,
            EmailVerificationCode.created_at >= daily_since,
        )
    )
    daily_count = int(daily_count_result.scalar_one() or 0)
    if daily_count >= settings.EMAIL_CODE_DAILY_LIMIT:
        raise ValueError(
            f"该邮箱 24 小时内最多发送 {settings.EMAIL_CODE_DAILY_LIMIT} 次验证码，请明天再试"
        )

    code = _generate_code()
    code_hash = _hash_email_code(clean_email, code, _REGISTER_PURPOSE, settings.EMAIL_CODE_SECRET)

    record = EmailVerificationCode(
        email=clean_email,
        purpose=_REGISTER_PURPOSE,
        code_hash=code_hash,
        expires_at=now + timedelta(minutes=settings.EMAIL_CODE_TTL_MINUTES),
    )
    session.add(record)
    await session.flush()

    await asyncio.to_thread(_send_email_code_sync, settings, clean_email, code)


async def _consume_registration_code(
    session: AsyncSession,
    email: str,
    code: str,
    settings: Settings,
) -> bool:
    clean_email = normalize_email(email)
    now = datetime.now(UTC)

    active_code_result = await session.execute(
        select(EmailVerificationCode)
        .where(
            EmailVerificationCode.email == clean_email,
            EmailVerificationCode.purpose == _REGISTER_PURPOSE,
            EmailVerificationCode.used_at.is_(None),
            EmailVerificationCode.expires_at >= now,
        )
        .order_by(desc(EmailVerificationCode.created_at))
        .limit(1)
    )
    active_code = active_code_result.scalar_one_or_none()
    if active_code is None:
        return False

    expected_hash = _hash_email_code(clean_email, code.strip(), _REGISTER_PURPOSE, settings.EMAIL_CODE_SECRET)
    if not hmac.compare_digest(active_code.code_hash, expected_hash):
        active_code.attempts += 1
        if active_code.attempts >= settings.EMAIL_CODE_MAX_ATTEMPTS:
            active_code.used_at = now
        await session.flush()
        return False

    active_code.used_at = now
    await session.flush()
    return True


async def register(
    session: AsyncSession,
    data: UserRegister,
    settings: Settings,
) -> tuple[UserResponse, AgentResponse]:
    """Create user + personal agent after successful email verification."""
    clean_username = data.username.strip()
    clean_agent_name = data.agent_name.strip()
    clean_email = normalize_email(data.email)

    if not clean_username or not clean_agent_name:
        raise ValueError("用户名和 Agent 名称不能为空")
    if not is_valid_email(clean_email):
        raise ValueError("邮箱格式不正确")

    pwd_ok, pwd_err = validate_password(data.password, settings.AUTH_PASSWORD_MIN_LENGTH)
    if not pwd_ok:
        raise ValueError(pwd_err)

    existing_username = await session.execute(
        select(func.count(User.id)).where(func.lower(User.username) == clean_username.lower())
    )
    if int(existing_username.scalar_one() or 0) > 0:
        raise ValueError("用户名已存在")

    existing_agent_name = await session.execute(
        select(func.count(Agent.id)).where(func.lower(Agent.name) == clean_agent_name.lower())
    )
    if int(existing_agent_name.scalar_one() or 0) > 0:
        raise ValueError("Agent 名称已存在")

    existing_email = await session.execute(select(User.id).where(User.email == clean_email))
    if existing_email.scalar_one_or_none() is not None:
        raise ValueError("邮箱已注册")

    code_ok = await _consume_registration_code(session, clean_email, data.verification_code, settings)
    if not code_ok:
        raise ValueError("验证码错误或已过期")

    user = User(
        username=clean_username,
        email=clean_email,
        password_hash=hash_password(data.password),
        email_verified=True,
    )
    session.add(user)

    try:
        await session.flush()
    except IntegrityError as exc:
        raise ValueError("注册失败，用户名或邮箱可能已存在") from exc

    agent = Agent(
        user_id=user.id,
        name=clean_agent_name,
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
        UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            email_verified=user.email_verified,
            created_at=user.created_at,
        ),
        AgentResponse(
            id=agent.id,
            user_id=agent.user_id,
            name=agent.name,
            personality=agent.personality,
            status=agent.status,
            created_at=agent.created_at,
        ),
    )


async def get_user_by_identifier(session: AsyncSession, identifier: str) -> User | None:
    clean_identifier = identifier.strip()
    if not clean_identifier:
        return None

    result = await session.execute(
        select(User).where(
            or_(
                User.username == clean_identifier,
                User.email == normalize_email(clean_identifier),
            )
        )
    )
    return result.scalar_one_or_none()


async def get_user(session: AsyncSession, user_id: int) -> UserResponse | None:
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        return None
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        email_verified=user.email_verified,
        created_at=user.created_at,
    )


async def get_agent(session: AsyncSession, agent_id: int) -> AgentResponse | None:
    result = await session.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if agent is None:
        return None
    return AgentResponse(
        id=agent.id,
        user_id=agent.user_id,
        name=agent.name,
        personality=agent.personality,
        status=agent.status,
        created_at=agent.created_at,
    )


async def get_agent_by_user(session: AsyncSession, user_id: int) -> AgentResponse | None:
    result = await session.execute(select(Agent).where(Agent.user_id == user_id))
    agent = result.scalar_one_or_none()
    if agent is None:
        return None
    return AgentResponse(
        id=agent.id,
        user_id=agent.user_id,
        name=agent.name,
        personality=agent.personality,
        status=agent.status,
        created_at=agent.created_at,
    )
