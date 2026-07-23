"""Request-scoped OpenAI credentials for multi-user application sessions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
from typing import Iterator


_SESSION_API_KEY: ContextVar[str | None] = ContextVar(
    "grounding_openai_api_key",
    default=None,
)


def resolve_openai_api_key(environment_name: str = "OPENAI_API_KEY") -> str | None:
    """Return the active session key, then fall back to the server environment."""
    session_key = _SESSION_API_KEY.get()
    if session_key:
        return session_key
    environment_key = os.environ.get(environment_name)
    return environment_key.strip() if environment_key else None


@contextmanager
def use_openai_api_key(api_key: str | None) -> Iterator[None]:
    """Scope a user-provided key to the current execution context."""
    cleaned = api_key.strip() if api_key else None
    token = _SESSION_API_KEY.set(cleaned or None)
    try:
        yield
    finally:
        _SESSION_API_KEY.reset(token)
