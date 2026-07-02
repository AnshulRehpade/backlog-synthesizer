"""Audit log for recording and retrieving sub-agent invocations.

This module provides the AuditLog class which records every sub-agent
invocation with a timestamp, agent name, input/output summaries, and
duration. Entries are retrievable by session ID in chronological order.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from backlog_synthesizer.models.memory import AuditEntry

_MAX_SUMMARY_LENGTH = 500
_SESSION_RETENTION_DAYS = 30
_SESSION_NOT_FOUND_MESSAGE = "Session not found or has expired"


class AuditLog:
    """Records and retrieves sub-agent invocation audit entries.

    Each entry is associated with a session ID and stored in chronological
    order. Sessions expire after 30 days of inactivity (matching the
    system-wide retention policy).
    """

    def __init__(self) -> None:
        """Initialize the audit log with empty storage."""
        self._entries: dict[str, list[AuditEntry]] = {}
        self._session_created_at: dict[str, datetime] = {}

    def _truncate(self, text: str) -> str:
        """Truncate text to the maximum summary length.

        Args:
            text: The text to truncate.

        Returns:
            The text truncated to 500 characters if it exceeds that length.
        """
        if len(text) > _MAX_SUMMARY_LENGTH:
            return text[:_MAX_SUMMARY_LENGTH]
        return text

    def _is_session_expired(self, session_id: str) -> bool:
        """Check whether a session has expired based on the 30-day retention.

        Args:
            session_id: The session identifier to check.

        Returns:
            True if the session has expired or does not exist.
        """
        if session_id not in self._session_created_at:
            return True
        created_at = self._session_created_at[session_id]
        expiration = created_at + timedelta(days=_SESSION_RETENTION_DAYS)
        return datetime.now(timezone.utc) > expiration

    def record(self, session_id: str, entry: AuditEntry) -> None:
        """Record a sub-agent invocation for a session.

        Truncates input_summary and output_summary to 500 characters if
        they exceed that length. Creates the session if it does not exist.

        Args:
            session_id: The session identifier to associate the entry with.
            entry: The audit entry to record.
        """
        truncated_entry = AuditEntry(
            timestamp=entry.timestamp,
            agent_name=entry.agent_name,
            input_summary=self._truncate(entry.input_summary),
            output_summary=self._truncate(entry.output_summary),
            duration_ms=entry.duration_ms,
        )

        if session_id not in self._entries:
            self._entries[session_id] = []
            self._session_created_at[session_id] = datetime.now(timezone.utc)

        self._entries[session_id].append(truncated_entry)

    def get_entries(self, session_id: str) -> list[AuditEntry]:
        """Retrieve audit entries for a session in chronological order.

        Args:
            session_id: The session identifier to retrieve entries for.

        Returns:
            A list of audit entries sorted by timestamp ascending.
            Returns an empty list if the session is not found or has expired.
        """
        if session_id not in self._entries or self._is_session_expired(session_id):
            return []

        entries = self._entries[session_id]
        return sorted(entries, key=lambda e: e.timestamp)

    def get_entries_with_status(
        self, session_id: str
    ) -> tuple[list[AuditEntry], str | None]:
        """Retrieve audit entries with a status message.

        Args:
            session_id: The session identifier to retrieve entries for.

        Returns:
            A tuple of (entries, message). If the session is valid and has
            entries, message is None. If the session is not found or has
            expired, entries is an empty list and message indicates the issue.
        """
        if session_id not in self._entries or self._is_session_expired(session_id):
            return [], _SESSION_NOT_FOUND_MESSAGE

        entries = sorted(self._entries[session_id], key=lambda e: e.timestamp)
        return entries, None
