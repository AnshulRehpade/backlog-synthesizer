"""Property-based tests for the Memory Engine.

Uses Hypothesis to verify behavioral properties of the AuditLog and ShortTermMemory
components of the Backlog Synthesizer system.
"""

import random
from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from backlog_synthesizer.memory.audit_log import AuditLog
from backlog_synthesizer.memory.short_term import ShortTermMemory
from backlog_synthesizer.models.memory import AuditEntry

# --- Strategies ---

# Strategy for non-empty session IDs
session_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=50,
).filter(lambda t: t.strip() != "")

# Strategy for non-empty keys
key_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=50,
).filter(lambda t: t.strip() != "")

# Strategy for arbitrary data payloads (JSON-compatible values)
data_payload_strategy = st.one_of(
    st.text(max_size=200),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.lists(st.integers(), max_size=10),
    st.dictionaries(
        keys=st.text(min_size=1, max_size=20).filter(lambda t: t.strip() != ""),
        values=st.one_of(st.integers(), st.text(max_size=50), st.booleans()),
        max_size=5,
    ),
)

# Strategy for AuditEntry instances with arbitrary timestamps
audit_entry_strategy = st.builds(
    AuditEntry,
    timestamp=st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31),
        timezones=st.just(timezone.utc),
    ),
    agent_name=st.text(min_size=1, max_size=50).filter(lambda t: t.strip() != ""),
    input_summary=st.text(min_size=1, max_size=500).filter(lambda t: t.strip() != ""),
    output_summary=st.text(min_size=1, max_size=500).filter(lambda t: t.strip() != ""),
    duration_ms=st.integers(min_value=0, max_value=100000),
)


# --- Property Tests ---


# Feature: backlog-synthesizer, Property 13: Audit log chronological ordering
class TestProperty13AuditLogChronologicalOrdering:
    """Verify audit log entries are returned sorted by timestamp ascending."""

    @given(
        entries=st.lists(audit_entry_strategy, min_size=1, max_size=20),
        session_id=session_id_strategy,
    )
    @settings(max_examples=100)
    def test_entries_returned_sorted_by_timestamp(
        self, entries: list[AuditEntry], session_id: str
    ) -> None:
        """
        For any set of audit entries recorded for a session (regardless of
        insertion order), retrieving the audit log for that session SHALL
        return entries sorted by timestamp in ascending order.

        **Validates: Requirements 6.7**
        """
        audit_log = AuditLog()

        # Shuffle entries to simulate arbitrary insertion order
        shuffled_entries = list(entries)
        random.shuffle(shuffled_entries)

        # Record entries in shuffled order
        for entry in shuffled_entries:
            audit_log.record(session_id, entry)

        # Retrieve and verify chronological ordering
        retrieved = audit_log.get_entries(session_id)

        assert len(retrieved) == len(entries)
        for i in range(len(retrieved) - 1):
            assert retrieved[i].timestamp <= retrieved[i + 1].timestamp, (
                f"Entry at index {i} (timestamp={retrieved[i].timestamp}) "
                f"is not <= entry at index {i+1} (timestamp={retrieved[i+1].timestamp})"
            )


# Feature: backlog-synthesizer, Property 14: Short-term memory round-trip by session ID
class TestProperty14ShortTermMemoryRoundTrip:
    """Verify short-term memory store/retrieve returns equal payload."""

    @given(
        session_id=session_id_strategy,
        key=key_strategy,
        data=data_payload_strategy,
    )
    @settings(max_examples=100)
    def test_store_retrieve_returns_equal_payload(
        self, session_id: str, key: str, data: object
    ) -> None:
        """
        For any session_id and data payload stored in Short_Term_Memory,
        retrieving by that session_id and key SHALL return a value equal
        to the originally stored payload.

        **Validates: Requirements 6.1**
        """
        memory = ShortTermMemory()

        # Store data
        memory.store(session_id, key, data)

        # Retrieve and verify equality
        retrieved = memory.retrieve(session_id, key)

        assert retrieved == data, (
            f"Retrieved value {retrieved!r} does not equal stored value {data!r} "
            f"for session_id={session_id!r}, key={key!r}"
        )
