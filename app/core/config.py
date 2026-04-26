from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Agent Matchmaking"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    DATABASE_URL: str = "sqlite+aiosqlite:///./data/matchmaking.db"

    CORS_ORIGINS: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "console"

    MAX_SIMULATION_BATCH: int = 200

    # LLM API for agent interactions (configure via .env)
    LLM_BASE_URL: str = ""
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "deepseek-chat"
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.7


def resolve_data_dir(database_url: str) -> Path:
    """Ensure the directory for a SQLite database URL exists."""
    if database_url.startswith("sqlite"):
        db_path = database_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
        db_file = Path(db_path)
        if not db_file.is_absolute():
            db_file = Path.cwd() / db_path
        db_file.parent.mkdir(parents=True, exist_ok=True)
    return Path.cwd()
