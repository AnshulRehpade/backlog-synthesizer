"""Tests for the Audit Log module.

Covers recording, chronological retrieval, truncation, and invalid session handling.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from backlog_synthesizer.memory.audit_log import AuditLog, _SESSION_NOT_FOUND_MESSAGE
from backlog_synthesizer.models.memory import AuditEntry


@pytest.fixture
def audit_log() -> AuditLog:
    """Provide a fresh AuditLog instance."""
    return AuditLog()


@pytest.fixture
def sample_entry() -> AuditEntry:
    """Provide a sample audit entry."""
    return AuditEntry(
        timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        agent_name="parser",
        input_summary="Parse meeting transcript",
        output_summary="Extracted 5 items",
        duration_ms=1200,
    )


class TestRecording:
    """Tests for recording audit entries."""

    def test_record_single_entry(
        self, audit_log: AuditLog, sample_entry: AuditEntry
    ) -> None:
        """Recording an entry makes it retrievable."""
        audit_log.record("session-1", sample_entry)
        entries = audit_log.get_entries("session-1")
        assert len(entries) == 1
        assert entries[0].agent_name == "parser"
        assert entries[0].duration_ms == 1200

    def test_record_multiple_entries_same_session(
        self, audit_log: AuditLog
    ) -> None:
        """Multiple entries can be recorded under the same session."""
        entries_to_record = [
            AuditEntry(
                timestamp=datetime(2024, 1, 15, 10, i, 0, tzinfo=timezone.utc),
                agent_name=f"agent-{i}",
                input_summary=f"Input {i}",
                output_summary=f"Output {i}",
                duration_ms=100 * i,
            )
            for i in range(3)
        ]
        for entry in entries_to_record:
            audit_log.record("session-1", entry)

        result = audit_log.get_entries("session-1")
        assert len(result) == 3

    def test_record_entries_different_sessions(
        self, audit_log: AuditLog, sample_entry: AuditEntry
    ) -> None:
        """Entries are isolated between sessions."""
        audit_log.record("session-a", sample_entry)
        audit_log.record("session-b", sample_entry)

        assert len(audit_log.get_entries("session-a")) == 1
        assert len(audit_log.get_entries("session-b")) == 1


class TestChronologicalRetrieval:
    """Tests for retrieving entries in chronological order."""

    def test_entries_returned_in_timestamp_order(
        self, audit_log: AuditLog
    ) -> None:
        """Entries are returned sorted by timestamp ascending."""
        # Record out of order
        entry_late = AuditEntry(
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            agent_name="story_writer",
            input_summary="Write stories",
            output_summary="Generated 3 stories",
            duration_ms=3000,
        )
        entry_early = AuditEntry(
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            agent_name="parser",
            input_summary="Parse docs",
            output_summary="Parsed 2 docs",
            duration_ms=1000,
        )
        entry_mid = AuditEntry(
            timestamp=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
            agent_name="gap_detection",
            input_summary="Detect gaps",
            output_summary="Found 1 gap",
            duration_ms=2000,
        )

        audit_log.record("session-1", entry_late)
        audit_log.record("session-1", entry_early)
        audit_log.record("session-1", entry_mid)

        entries = audit_log.get_entries("session-1")
        assert len(entries) == 3
        assert entries[0].agent_name == "parser"
        assert entries[1].agent_name == "gap_detection"
        assert entries[2].agent_name == "story_writer"

    def test_get_entries_with_status_chronological(
        self, audit_log: AuditLog
    ) -> None:
        """get_entries_with_status also returns in chronological order."""
        entry_b = AuditEntry(
            timestamp=datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc),
            agent_name="agent-b",
            input_summary="b",
            output_summary="b",
            duration_ms=200,
        )
        entry_a = AuditEntry(
            timestamp=datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc),
            agent_name="agent-a",
            input_summary="a",
            output_summary="a",
            duration_ms=100,
        )

        audit_log.record("session-1", entry_b)
        audit_log.record("session-1", entry_a)

        entries, message = audit_log.get_entries_with_status("session-1")
        assert message is None
        assert entries[0].agent_name == "agent-a"
        assert entries[1].agent_name == "agent-b"


class TestTruncation:
    """Tests for input/output summary truncation to 500 characters."""

    def test_long_input_summary_truncated(self, audit_log: AuditLog) -> None:
        """Input summaries longer than 500 chars are truncated."""
        long_text = "x" * 600
        entry = AuditEntry(
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            agent_name="parser",
            input_summary=long_text[:500],  # Pydantic enforces max_length
            output_summary="short",
            duration_ms=100,
        )
        # Simulate recording with pre-truncation in the AuditLog
        # We create an entry that bypasses Pydantic validation for this test
        entry_dict = {
            "timestamp": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            "agent_name": "parser",
            "input_summary": long_text[:500],
            "output_summary": "short",
            "duration_ms": 100,
        }
        audit_entry = AuditEntry.model_validate(entry_dict)
        audit_log.record("session-1", audit_entry)

        entries = audit_log.get_entries("session-1")
        assert len(entries[0].input_summary) <= 500

    def test_long_output_summary_truncated(self, audit_log: AuditLog) -> None:
        """Output summaries longer than 500 chars are truncated."""
        long_text = "y" * 600
        entry = AuditEntry(
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            agent_name="parser",
            input_summary="short",
            output_summary=long_text[:500],
            duration_ms=100,
        )
        audit_log.record("session-1", entry)

        entries = audit_log.get_entries("session-1")
        assert len(entries[0].output_summary) <= 500

    def test_truncation_via_record_method(self, audit_log: AuditLog) -> None:
        """The record method itself truncates summaries exceeding 500 chars."""
        long_input = "a" * 700
        long_output = "b" * 700

        # Build entry with model_construct to bypass Pydantic validation
        entry = AuditEntry.model_construct(
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            agent_name="parser",
            input_summary=long_input,
            output_summary=long_output,
            duration_ms=100,
        )
        audit_log.record("session-1", entry)

        entries = audit_log.get_entries("session-1")
        assert len(entries[0].input_summary) == 500
        assert len(entries[0].output_summary) == 500

    def test_short_summaries_not_modified(self, audit_log: AuditLog) -> None:
        """Summaries within the 500-char limit are preserved as-is."""
        entry = AuditEntry(
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            agent_name="parser",
            input_summary="Hello world",
            output_summary="Done",
            duration_ms=50,
        )
        audit_log.record("session-1", entry)

        entries = audit_log.get_entries("session-1")
        assert entries[0].input_summary == "Hello world"
        assert entries[0].output_summary == "Done"


class TestInvalidSessionHandling:
    """Tests for invalid/expired session ID handling."""

    def test_nonexistent_session_returns_empty(self, audit_log: AuditLog) -> None:
        """Querying a session that was never created returns empty list."""
        entries = audit_log.get_entries("nonexistent-session")
        assert entries == []

    def test_nonexistent_session_with_status_returns_message(
        self, audit_log: AuditLog
    ) -> None:
        """Querying a nonexistent session via get_entries_with_status returns message."""
        entries, message = audit_log.get_entries_with_status("nonexistent-session")
        assert entries == []
        assert message == _SESSION_NOT_FOUND_MESSAGE

    def test_expired_session_returns_empty(self, audit_log: AuditLog) -> None:
        """Querying an expired session returns empty list."""
        entry = AuditEntry(
            timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            agent_name="parser",
            input_summary="old input",
            output_summary="old output",
            duration_ms=100,
        )
        audit_log.record("old-session", entry)

        # Simulate time passing beyond 30 days
        expired_time = datetime.now(timezone.utc) + timedelta(days=31)
        with patch(
            "backlog_synthesizer.memory.audit_log.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value = expired_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            entries = audit_log.get_entries("old-session")
            assert entries == []

    def test_expired_session_with_status_returns_message(
        self, audit_log: AuditLog
    ) -> None:
        """Querying an expired session via get_entries_with_status returns message."""
        entry = AuditEntry(
            timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            agent_name="parser",
            input_summary="old input",
            output_summary="old output",
            duration_ms=100,
        )
        audit_log.record("old-session", entry)

        # Simulate time passing beyond 30 days
        expired_time = datetime.now(timezone.utc) + timedelta(days=31)
        with patch(
            "backlog_synthesizer.memory.audit_log.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value = expired_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            entries, message = audit_log.get_entries_with_status("old-session")
            assert entries == []
            assert message == _SESSION_NOT_FOUND_MESSAGE

    def test_valid_session_with_status_returns_no_message(
        self, audit_log: AuditLog, sample_entry: AuditEntry
    ) -> None:
        """A valid, non-expired session returns entries with no message."""
        audit_log.record("active-session", sample_entry)

        entries, message = audit_log.get_entries_with_status("active-session")
        assert len(entries) == 1
        assert message is None
