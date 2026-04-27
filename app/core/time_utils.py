from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

UTC_PLUS_8 = timezone(timedelta(hours=8))


def now_utc8() -> datetime:
    return datetime.now(UTC_PLUS_8)


def ensure_utc8(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Existing SQLite rows may be timezone-naive; treat them as UTC for back-compat.
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC_PLUS_8)


def format_utc8(dt: datetime | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    localized = ensure_utc8(dt)
    if localized is None:
        return ""
    return localized.strftime(fmt)
