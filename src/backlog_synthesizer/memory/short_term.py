"""Short-Term Memory implementation for the Backlog Synthesizer system.

Provides in-session state storage accessible by session ID. Supports a primary
store backend (e.g., Redis) with automatic fallback to an in-process Python
dictionary when the primary store is unavailable. Logs a warning to the Audit Log
when fallback is activated.

Validates: Requirements 6.1, 6.6
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ShortTermMemoryStore(Protocol):
    """Protocol defining the primary store interface for short-term memory.

    Implementations may use Redis, Memcached, or any other session-scoped
    key-value store. Raises Exception on unavailability so the fallback
    mechanism can activate.
    """

    def store(self, session_id: str, key: str, data: Any) -> None:
        """Store data under a session-scoped key.

        Args:
            session_id: The session identifier.
            key: The key within the session namespace.
            data: The data to store.

        Raises:
            Exception: If the store is unavailable or write fails.
        """
        ...

    def retrieve(self, session_id: str, key: str) -> Any:
        """Retrieve data by session ID and key.

        Args:
            session_id: The session identifier.
            key: The key within the session namespace.

        Returns:
            The stored data, or None if not found.

        Raises:
            Exception: If the store is unavailable or read fails.
        """
        ...


class AuditLogger(Protocol):
    """Protocol for audit log warning callbacks.

    Used to notify the audit system when fallback memory is activated.
    """

    def log_warning(self, session_id: str, message: str) -> None:
        """Log a warning message associated with a session.

        Args:
            session_id: The session identifier.
            message: The warning message to log.
        """
        ...


class ShortTermMemory:
    """In-session state store accessible by session ID.

    Stores all intermediate results from the current session in short-term
    memory. Supports a primary store backend with automatic fallback to an
    in-process Python dictionary when the primary store is unavailable.

    When fallback is activated, a warning is logged to the Audit Log.

    Validates: Requirements 6.1, 6.6
    """

    def __init__(
        self,
        primary_store: ShortTermMemoryStore | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialize short-term memory with optional primary store and audit logger.

        Args:
            primary_store: The primary backing store (e.g., Redis). If None,
                the fallback dict is used immediately.
            audit_logger: Optional audit logger for fallback warnings.
        """
        self._primary_store = primary_store
        self._audit_logger = audit_logger
        self._fallback: dict[str, dict[str, Any]] = {}
        self._using_fallback = primary_store is None
        self._fallback_warned_sessions: set[str] = set()

        if self._using_fallback:
            logger.warning(
                "ShortTermMemory initialized without primary store; "
                "using in-process dict fallback."
            )

    @property
    def using_fallback(self) -> bool:
        """Whether the memory is currently operating in fallback mode."""
        return self._using_fallback

    def store(self, session_id: str, key: str, data: Any) -> None:
        """Store data under a session-scoped key.

        Attempts to write to the primary store first. If the primary store
        is unavailable, falls back to an in-process Python dictionary and
        logs a warning to the Audit Log.

        Args:
            session_id: The session identifier.
            key: The key within the session namespace.
            data: The data to store (any serializable value).

        Validates: Requirements 6.1, 6.6
        """
        if not self._using_fallback and self._primary_store is not None:
            try:
                self._primary_store.store(session_id, key, data)
                return
            except Exception as exc:
                self._activate_fallback(session_id, exc)

        # Fallback: store in-process dict
        if session_id not in self._fallback:
            self._fallback[session_id] = {}
        self._fallback[session_id][key] = data

    def retrieve(self, session_id: str, key: str) -> Any:
        """Retrieve data by session ID and key.

        Attempts to read from the primary store first. If the primary store
        is unavailable, falls back to the in-process Python dictionary.

        Args:
            session_id: The session identifier.
            key: The key within the session namespace.

        Returns:
            The stored data, or None if the key does not exist.

        Validates: Requirements 6.1, 6.6
        """
        if not self._using_fallback and self._primary_store is not None:
            try:
                return self._primary_store.retrieve(session_id, key)
            except Exception as exc:
                self._activate_fallback(session_id, exc)

        # Fallback: read from in-process dict
        session_data = self._fallback.get(session_id)
        if session_data is None:
            return None
        return session_data.get(key)

    def _activate_fallback(self, session_id: str, exc: Exception) -> None:
        """Activate fallback mode and log warning.

        Args:
            session_id: The session that triggered the fallback.
            exc: The exception from the primary store.
        """
        self._using_fallback = True
        warning_msg = (
            f"Short-term memory primary store unavailable "
            f"(error: {exc}); falling back to in-process Python dictionary."
        )
        logger.warning(warning_msg)

        # Log to audit log (once per session to avoid spam)
        if session_id not in self._fallback_warned_sessions:
            self._fallback_warned_sessions.add(session_id)
            if self._audit_logger is not None:
                self._audit_logger.log_warning(session_id, warning_msg)
