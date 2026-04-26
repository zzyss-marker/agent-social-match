from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.core.config import Settings


def setup_logging(settings: Settings) -> None:
    """Configure structlog with console or JSON rendering."""
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    if settings.LOG_FORMAT == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            timestamper,
            structlog.dev.set_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.LOG_LEVEL)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Suppress noisy SQLAlchemy logging in non-DEBUG mode
    if settings.LOG_LEVEL != "DEBUG":
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
