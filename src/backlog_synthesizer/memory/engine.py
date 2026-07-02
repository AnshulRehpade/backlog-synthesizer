"""Memory Engine facade wiring Short-Term Memory, Long-Term Memory, and Audit Log.

Provides a unified interface for session state storage, semantic search indexing,
and audit logging. Implements the AuditLogger protocol so it can serve as the
fallback-warning logger for ShortTermMemory.

Validates: Requirements 6.1, 6.2, 6.4, 6.7
"""

from __future__ import annotations

import logging
from typing import Any

from backlog_synthesizer.memory.audit_log import AuditLog
from backlog_synthesizer.memory.long_term import LongTermMemory
from backlog_synthesizer.memory.short_term import ShortTermMemory
from backlog_synthesizer.models.memory import AuditEntry
from backlog_synthesizer.tools.interfaces import SearchResult

logger = logging.getLogger(__name__)


class MemoryEngine:
    """Facade coordinating Short-Term Memory, Long-Term Memory, and Audit Log.

    Provides a single entry point for:
    - Storing/retrieving intermediate session results (Short-Term Memory)
    - Indexing items for semantic search (Long-Term Memory)
    - Recording and querying audit log entries (Audit Log)

    Also implements the AuditLogger protocol expected by ShortTermMemory,
    allowing the engine to log fallback warnings into the audit system.

    Args:
        short_term: ShortTermMemory instance for session-scoped storage.
        long_term: LongTermMemory instance for vector-backed semantic search.
        audit_log: AuditLog instance for recording sub-agent invocations.
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        audit_log: AuditLog,
    ) -> None:
        self._short_term = short_term
        self._long_term = long_term
        self._audit_log = audit_log

    # ──────────────────────────────────────────────
    # Short-Term Memory delegation (Requirement 6.1)
    # ──────────────────────────────────────────────

    def store_intermediate(self, session_id: str, key: str, data: Any) -> None:
        """Store an intermediate result in short-term memory.

        Args:
            session_id: The session identifier.
            key: The key within the session namespace.
            data: The data to store.

        Validates: Requirement 6.1
        """
        self._short_term.store(session_id, key, data)

    def retrieve_intermediate(self, session_id: str, key: str) -> Any:
        """Retrieve an intermediate result from short-term memory.

        Args:
            session_id: The session identifier.
            key: The key within the session namespace.

        Returns:
            The stored data, or None if not found.

        Validates: Requirement 6.1
        """
        return self._short_term.retrieve(session_id, key)

    # ──────────────────────────────────────────────
    # Long-Term Memory delegation (Requirement 6.2)
    # ──────────────────────────────────────────────

    def store_for_search(self, session_id: str, items: list[dict[str, Any]]) -> None:
        """Index items in long-term memory for semantic search.

        Each item dict must contain at least `item_id` and `content` keys.
        The session_id is added to the item metadata for traceability.

        Args:
            session_id: The session identifier that produced these items.
            items: List of dicts with at least `item_id` and `content` keys.
                Additional keys are stored as metadata.

        Validates: Requirement 6.2
        """
        for item in items:
            item_id = item["item_id"]
            content = item["content"]
            metadata: dict[str, Any] = {
                k: v for k, v in item.items() if k not in ("item_id", "content")
            }
            metadata["session_id"] = session_id
            self._long_term.store_item(item_id, content, metadata)

    def search_similar(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search for items semantically similar to the query.

        Args:
            query: Text to search for similar items.
            top_k: Maximum number of results to return. Defaults to 10.

        Returns:
            List of SearchResult objects ordered by similarity score (descending).

        Validates: Requirement 6.2
        """
        return self._long_term.search_similar(query, top_k)

    # ──────────────────────────────────────────────
    # Audit Log delegation (Requirements 6.4, 6.7)
    # ──────────────────────────────────────────────

    def log_action(self, session_id: str, entry: AuditEntry) -> None:
        """Record a sub-agent invocation in the audit log.

        Args:
            session_id: The session identifier.
            entry: The audit entry to record.

        Validates: Requirement 6.4
        """
        self._audit_log.record(session_id, entry)

    def get_audit_log(self, session_id: str) -> list[AuditEntry]:
        """Retrieve all audit log entries for a session in chronological order.

        Args:
            session_id: The session identifier.

        Returns:
            A list of audit entries sorted by timestamp ascending.
            Returns an empty list if no entries exist for the session.

        Validates: Requirement 6.7
        """
        return self._audit_log.get_entries(session_id)

    # ──────────────────────────────────────────────
    # AuditLogger protocol implementation
    # ──────────────────────────────────────────────

    def log_warning(self, session_id: str, message: str) -> None:
        """Log a warning message (satisfies the AuditLogger protocol).

        Used by ShortTermMemory to report fallback activation warnings.

        Args:
            session_id: The session identifier.
            message: The warning message to log.
        """
        logger.warning("Memory fallback warning [session=%s]: %s", session_id, message)
