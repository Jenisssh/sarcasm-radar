"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import sys
from typing import cast

import structlog
from structlog.stdlib import BoundLogger

_configured = False


def configure(level: str = "INFO") -> None:
    """Idempotent one-time structlog configuration."""
    global _configured
    if _configured:
        return

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> BoundLogger:
    """Return a configured structlog logger."""
    configure()
    return cast(BoundLogger, structlog.get_logger(name))
