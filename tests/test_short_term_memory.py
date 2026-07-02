"""Tests for ShortTermMemory covering normal operation and fallback behavior.

Validates: Requirements 6.1, 6.6
"""

from __future__ import annotations

from typing import Any

import pytest

from backlog_synthesizer.memory.short_term import (
    AuditLogger,
    ShortTermMemory,
    ShortTermMemoryStore,
)


# --- Test doubles ---


class InMemoryPrimaryStore:
    """A simple in-memory primary store for testing normal operation."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def store(self, session_id: str, key: str, data: Any) -> None:
        if session_id not in self._data:
            self._data[session_id] = {}
        self._data[session_id][key] = data

    def retrieve(self, session_id: str, key: str) -> Any:
        session_data = self._data.get(session_id)
        if session_data is None:
            return None
        return session_data.get(key)


class FailingPrimaryStore:
    """A primary store that always raises to simulate unavailability."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error or ConnectionError("Redis connection refused")

    def store(self, session_id: str, key: str, data: Any) -> None:
        raise self._error

    def retrieve(self, session_id: str, key: str) -> Any:
        raise self._error


class FakeAuditLogger:
    """Records audit log warnings for test assertions."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, str]] = []

    def log_warning(self, session_id: str, message: str) -> None:
        self.warnings.append((session_id, message))


# --- Fixtures ---


@pytest.fixture
def primary_store() -> InMemoryPrimaryStore:
    return InMemoryPrimaryStore()


@pytest.fixture
def failing_store() -> FailingPrimaryStore:
    return FailingPrimaryStore()


@pytest.fixture
def audit_logger() -> FakeAuditLogger:
    return FakeAuditLogger()


# --- Normal operation tests ---


class TestShortTermMemoryNormalOperation:
    """Tests for ShortTermMemory with a functioning primary store."""

    def test_store_and_retrieve(self, primary_store: InMemoryPrimaryStore) -> None:
        """Store data and retrieve it by session ID and key."""
        memory = ShortTermMemory(primary_store=primary_store)

        memory.store("session-1", "extraction_result", {"items": [1, 2, 3]})
        result = memory.retrieve("session-1", "extraction_result")

        assert result == {"items": [1, 2, 3]}

    def test_retrieve_nonexistent_key_returns_none(
        self, primary_store: InMemoryPrimaryStore
    ) -> None:
        """Retrieving a non-existent key returns None."""
        memory = ShortTermMemory(primary_store=primary_store)

        result = memory.retrieve("session-1", "nonexistent")

        assert result is None

    def test_separate_sessions_isolated(
        self, primary_store: InMemoryPrimaryStore
    ) -> None:
        """Data stored under different session IDs is isolated."""
        memory = ShortTermMemory(primary_store=primary_store)

        memory.store("session-a", "key", "value-a")
        memory.store("session-b", "key", "value-b")

        assert memory.retrieve("session-a", "key") == "value-a"
        assert memory.retrieve("session-b", "key") == "value-b"

    def test_overwrite_existing_key(
        self, primary_store: InMemoryPrimaryStore
    ) -> None:
        """Storing with the same key overwrites previous value."""
        memory = ShortTermMemory(primary_store=primary_store)

        memory.store("session-1", "status", "in_progress")
        memory.store("session-1", "status", "completed")

        assert memory.retrieve("session-1", "status") == "completed"

    def test_not_using_fallback_with_primary_store(
        self, primary_store: InMemoryPrimaryStore
    ) -> None:
        """When primary store works, fallback is not active."""
        memory = ShortTermMemory(primary_store=primary_store)

        assert memory.using_fallback is False

    def test_stores_complex_data_types(
        self, primary_store: InMemoryPrimaryStore
    ) -> None:
        """Supports storing complex nested data structures."""
        memory = ShortTermMemory(primary_store=primary_store)
        complex_data = {
            "items": [{"type": "decision", "text": "Use Python"}],
            "metadata": {"count": 1},
        }

        memory.store("session-1", "result", complex_data)

        assert memory.retrieve("session-1", "result") == complex_data


# --- Fallback behavior tests ---


class TestShortTermMemoryFallback:
    """Tests for ShortTermMemory fallback when primary store is unavailable."""

    def test_fallback_on_store_failure(
        self, failing_store: FailingPrimaryStore, audit_logger: FakeAuditLogger
    ) -> None:
        """Falls back to dict when primary store raises on store."""
        memory = ShortTermMemory(
            primary_store=failing_store, audit_logger=audit_logger
        )

        memory.store("session-1", "key", "value")
        result = memory.retrieve("session-1", "key")

        assert result == "value"
        assert memory.using_fallback is True

    def test_fallback_on_retrieve_failure(
        self, failing_store: FailingPrimaryStore, audit_logger: FakeAuditLogger
    ) -> None:
        """Falls back to dict when primary store raises on retrieve."""
        memory = ShortTermMemory(
            primary_store=failing_store, audit_logger=audit_logger
        )

        # Store will fail and activate fallback, then store in dict
        memory.store("session-1", "key", "value")
        # Retrieve should use fallback dict
        result = memory.retrieve("session-1", "key")

        assert result == "value"

    def test_fallback_logs_warning_to_audit_log(
        self, failing_store: FailingPrimaryStore, audit_logger: FakeAuditLogger
    ) -> None:
        """Logs warning to Audit Log when fallback is activated."""
        memory = ShortTermMemory(
            primary_store=failing_store, audit_logger=audit_logger
        )

        memory.store("session-1", "key", "value")

        assert len(audit_logger.warnings) == 1
        session_id, message = audit_logger.warnings[0]
        assert session_id == "session-1"
        assert "unavailable" in message.lower()
        assert "fallback" in message.lower() or "dict" in message.lower()

    def test_fallback_warning_logged_once_per_session(
        self, failing_store: FailingPrimaryStore, audit_logger: FakeAuditLogger
    ) -> None:
        """Audit warning is logged only once per session, not on every operation."""
        memory = ShortTermMemory(
            primary_store=failing_store, audit_logger=audit_logger
        )

        memory.store("session-1", "key1", "value1")
        memory.store("session-1", "key2", "value2")

        # Only one warning for session-1
        session_warnings = [
            w for w in audit_logger.warnings if w[0] == "session-1"
        ]
        assert len(session_warnings) == 1

    def test_fallback_without_primary_store(
        self, audit_logger: FakeAuditLogger
    ) -> None:
        """When no primary store is provided, uses fallback immediately."""
        memory = ShortTermMemory(primary_store=None, audit_logger=audit_logger)

        assert memory.using_fallback is True

        memory.store("session-1", "key", "value")
        assert memory.retrieve("session-1", "key") == "value"

    def test_fallback_retrieve_nonexistent_session_returns_none(
        self, failing_store: FailingPrimaryStore
    ) -> None:
        """Fallback returns None for non-existent session."""
        memory = ShortTermMemory(primary_store=failing_store)

        # Trigger fallback
        memory.store("session-1", "key", "value")

        assert memory.retrieve("nonexistent-session", "key") is None

    def test_fallback_retrieve_nonexistent_key_returns_none(
        self, failing_store: FailingPrimaryStore
    ) -> None:
        """Fallback returns None for non-existent key in existing session."""
        memory = ShortTermMemory(primary_store=failing_store)

        memory.store("session-1", "existing-key", "value")

        assert memory.retrieve("session-1", "nonexistent-key") is None

    def test_fallback_without_audit_logger(
        self, failing_store: FailingPrimaryStore
    ) -> None:
        """Fallback works gracefully even without an audit logger."""
        memory = ShortTermMemory(primary_store=failing_store, audit_logger=None)

        # Should not raise
        memory.store("session-1", "key", "value")
        assert memory.retrieve("session-1", "key") == "value"
        assert memory.using_fallback is True

    def test_fallback_persists_across_operations(
        self, audit_logger: FakeAuditLogger,
    ) -> None:
        """Once fallback is activated, subsequent operations use it."""

        class FailOnceThenSucceedStore:
            """Fails on first call, then succeeds."""

            def __init__(self) -> None:
                self._call_count = 0

            def store(self, session_id: str, key: str, data: Any) -> None:
                self._call_count += 1
                if self._call_count == 1:
                    raise ConnectionError("First call fails")
                # Would succeed but should not be called after fallback

            def retrieve(self, session_id: str, key: str) -> Any:
                return "from-primary"

        store = FailOnceThenSucceedStore()
        memory = ShortTermMemory(primary_store=store, audit_logger=audit_logger)

        # First store triggers fallback
        memory.store("session-1", "key1", "value1")
        # Second store should use fallback, not primary
        memory.store("session-1", "key2", "value2")

        assert memory.retrieve("session-1", "key1") == "value1"
        assert memory.retrieve("session-1", "key2") == "value2"
        assert memory.using_fallback is True
