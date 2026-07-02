"""Tests for the MemoryEngine facade.

Validates that the MemoryEngine correctly delegates to ShortTermMemory,
LongTermMemory, and AuditLog, and implements the AuditLogger protocol.
"""

from datetime import datetime, timezone

import pytest

from backlog_synthesizer.memory.audit_log import AuditLog
from backlog_synthesizer.memory.engine import MemoryEngine
from backlog_synthesizer.memory.long_term import LongTermMemory
from backlog_synthesizer.memory.short_term import AuditLogger, ShortTermMemory
from backlog_synthesizer.models.memory import AuditEntry
from backlog_synthesizer.tools.interfaces import SearchResult


# ──────────────────────────────────────────────────────
# Fake tool implementations for testing
# ──────────────────────────────────────────────────────


class FakeEmbeddingTool:
    """Fake embedding tool that returns a fixed-length vector from the text hash."""

    def generate_embedding(self, text: str) -> list[float]:
        # Simple deterministic embedding: normalized character codes
        return [float(ord(c) % 100) / 100.0 for c in text[:8].ljust(8)]


class FakeVectorSearchTool:
    """In-memory fake vector store for testing."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[list[float], dict]] = {}

    def store(self, item_id: str, embedding: list[float], metadata: dict) -> None:
        self._store[item_id] = (embedding, metadata)

    def query_similar(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        # Return all stored items with a fake score (simplistic matching)
        results = []
        for item_id, (stored_emb, metadata) in self._store.items():
            # Simple dot-product-like similarity score
            score = sum(a * b for a, b in zip(embedding, stored_emb))
            results.append(SearchResult(item_id=item_id, score=score, metadata=metadata))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ──────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────


@pytest.fixture
def fake_embedding_tool() -> FakeEmbeddingTool:
    return FakeEmbeddingTool()


@pytest.fixture
def fake_vector_tool() -> FakeVectorSearchTool:
    return FakeVectorSearchTool()


@pytest.fixture
def short_term() -> ShortTermMemory:
    """ShortTermMemory using the in-process fallback dict."""
    return ShortTermMemory()


@pytest.fixture
def long_term(
    fake_embedding_tool: FakeEmbeddingTool,
    fake_vector_tool: FakeVectorSearchTool,
) -> LongTermMemory:
    return LongTermMemory(
        embedding_tool=fake_embedding_tool,
        vector_search_tool=fake_vector_tool,
    )


@pytest.fixture
def audit_log() -> AuditLog:
    return AuditLog()


@pytest.fixture
def engine(
    short_term: ShortTermMemory,
    long_term: LongTermMemory,
    audit_log: AuditLog,
) -> MemoryEngine:
    return MemoryEngine(
        short_term=short_term,
        long_term=long_term,
        audit_log=audit_log,
    )


# ──────────────────────────────────────────────────────
# Tests: Short-Term Memory delegation (Requirement 6.1)
# ──────────────────────────────────────────────────────


class TestStoreIntermediate:
    """Tests for store_intermediate and retrieve_intermediate."""

    def test_store_and_retrieve_basic(self, engine: MemoryEngine) -> None:
        """Storing a value and retrieving it returns the same value."""
        engine.store_intermediate("session-1", "extraction_result", {"items": [1, 2, 3]})
        result = engine.retrieve_intermediate("session-1", "extraction_result")
        assert result == {"items": [1, 2, 3]}

    def test_retrieve_nonexistent_key_returns_none(self, engine: MemoryEngine) -> None:
        """Retrieving a key that was never stored returns None."""
        result = engine.retrieve_intermediate("session-1", "missing_key")
        assert result is None

    def test_multiple_sessions_isolated(self, engine: MemoryEngine) -> None:
        """Data stored under one session is not visible in another."""
        engine.store_intermediate("session-a", "key1", "value-a")
        engine.store_intermediate("session-b", "key1", "value-b")

        assert engine.retrieve_intermediate("session-a", "key1") == "value-a"
        assert engine.retrieve_intermediate("session-b", "key1") == "value-b"

    def test_overwrite_existing_key(self, engine: MemoryEngine) -> None:
        """Storing to the same key overwrites the previous value."""
        engine.store_intermediate("session-1", "key", "old")
        engine.store_intermediate("session-1", "key", "new")
        assert engine.retrieve_intermediate("session-1", "key") == "new"


# ──────────────────────────────────────────────────────
# Tests: Long-Term Memory delegation (Requirement 6.2)
# ──────────────────────────────────────────────────────


class TestStoreForSearch:
    """Tests for store_for_search and search_similar."""

    def test_store_and_search_items(self, engine: MemoryEngine) -> None:
        """Items stored for search are retrievable via semantic search."""
        items = [
            {"item_id": "story-1", "content": "User login flow"},
            {"item_id": "story-2", "content": "Password reset feature"},
        ]
        engine.store_for_search("session-1", items)

        results = engine.search_similar("login", top_k=5)
        assert len(results) == 2
        item_ids = {r.item_id for r in results}
        assert "story-1" in item_ids
        assert "story-2" in item_ids

    def test_session_id_in_metadata(
        self,
        engine: MemoryEngine,
        fake_vector_tool: FakeVectorSearchTool,
    ) -> None:
        """Stored items include session_id in their metadata."""
        items = [{"item_id": "item-1", "content": "Some content"}]
        engine.store_for_search("session-42", items)

        # Verify the stored metadata in the fake vector tool
        stored_embedding, stored_metadata = fake_vector_tool._store["item-1"]
        assert stored_metadata["session_id"] == "session-42"

    def test_extra_metadata_preserved(
        self,
        engine: MemoryEngine,
        fake_vector_tool: FakeVectorSearchTool,
    ) -> None:
        """Additional keys in item dicts are preserved as metadata."""
        items = [
            {
                "item_id": "item-1",
                "content": "Some content",
                "priority": "high",
                "category": "auth",
            }
        ]
        engine.store_for_search("session-1", items)

        _, stored_metadata = fake_vector_tool._store["item-1"]
        assert stored_metadata["priority"] == "high"
        assert stored_metadata["category"] == "auth"

    def test_search_similar_respects_top_k(self, engine: MemoryEngine) -> None:
        """search_similar returns at most top_k results."""
        items = [
            {"item_id": f"item-{i}", "content": f"content {i}"} for i in range(5)
        ]
        engine.store_for_search("session-1", items)

        results = engine.search_similar("content", top_k=2)
        assert len(results) <= 2


# ──────────────────────────────────────────────────────
# Tests: Audit Log delegation (Requirements 6.4, 6.7)
# ──────────────────────────────────────────────────────


class TestAuditLog:
    """Tests for log_action and get_audit_log."""

    def _make_entry(
        self,
        agent_name: str = "test-agent",
        duration_ms: int = 100,
        timestamp: datetime | None = None,
    ) -> AuditEntry:
        return AuditEntry(
            timestamp=timestamp or datetime.now(timezone.utc),
            agent_name=agent_name,
            input_summary="test input",
            output_summary="test output",
            duration_ms=duration_ms,
        )

    def test_log_and_retrieve_entries(self, engine: MemoryEngine) -> None:
        """Logged entries are retrievable via get_audit_log."""
        entry = self._make_entry()
        engine.log_action("session-1", entry)

        entries = engine.get_audit_log("session-1")
        assert len(entries) == 1
        assert entries[0].agent_name == "test-agent"
        assert entries[0].duration_ms == 100

    def test_entries_returned_in_chronological_order(
        self, engine: MemoryEngine
    ) -> None:
        """Audit entries are returned sorted by timestamp ascending."""
        entry1 = self._make_entry(
            agent_name="first",
            timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        )
        entry2 = self._make_entry(
            agent_name="second",
            timestamp=datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
        )
        # Log in reverse order
        engine.log_action("session-1", entry2)
        engine.log_action("session-1", entry1)

        entries = engine.get_audit_log("session-1")
        assert entries[0].agent_name == "first"
        assert entries[1].agent_name == "second"

    def test_get_audit_log_empty_session(self, engine: MemoryEngine) -> None:
        """get_audit_log returns empty list for unknown sessions."""
        entries = engine.get_audit_log("nonexistent-session")
        assert entries == []

    def test_sessions_are_isolated(self, engine: MemoryEngine) -> None:
        """Audit entries from different sessions are isolated."""
        entry_a = self._make_entry(agent_name="agent-a")
        entry_b = self._make_entry(agent_name="agent-b")

        engine.log_action("session-a", entry_a)
        engine.log_action("session-b", entry_b)

        entries_a = engine.get_audit_log("session-a")
        entries_b = engine.get_audit_log("session-b")

        assert len(entries_a) == 1
        assert entries_a[0].agent_name == "agent-a"
        assert len(entries_b) == 1
        assert entries_b[0].agent_name == "agent-b"


# ──────────────────────────────────────────────────────
# Tests: AuditLogger protocol implementation
# ──────────────────────────────────────────────────────


class TestAuditLoggerProtocol:
    """Tests that MemoryEngine satisfies the AuditLogger protocol."""

    def test_satisfies_audit_logger_protocol(self, engine: MemoryEngine) -> None:
        """MemoryEngine has the log_warning method required by AuditLogger."""
        assert hasattr(engine, "log_warning")
        assert callable(engine.log_warning)

    def test_log_warning_does_not_raise(self, engine: MemoryEngine) -> None:
        """log_warning runs without error."""
        # Should not raise
        engine.log_warning("session-1", "Fallback activated")

    def test_engine_usable_as_audit_logger_for_short_term(
        self,
        long_term: LongTermMemory,
        audit_log: AuditLog,
    ) -> None:
        """MemoryEngine can be passed as audit_logger to ShortTermMemory."""
        engine = MemoryEngine(
            short_term=ShortTermMemory(),  # temporary, replaced below
            long_term=long_term,
            audit_log=audit_log,
        )
        # Create a ShortTermMemory that uses the engine as its audit logger
        stm = ShortTermMemory(audit_logger=engine)
        # Store and retrieve should work fine
        stm.store("session-1", "key", "value")
        assert stm.retrieve("session-1", "key") == "value"


# ──────────────────────────────────────────────────────
# Integration test: full workflow
# ──────────────────────────────────────────────────────


class TestIntegration:
    """Integration tests exercising multiple MemoryEngine methods together."""

    def test_full_session_workflow(self, engine: MemoryEngine) -> None:
        """Simulate a full session: store intermediate, index for search, log actions."""
        session_id = "integration-session"

        # Store intermediate results
        engine.store_intermediate(session_id, "raw_extraction", {"themes": ["auth"]})
        engine.store_intermediate(session_id, "gap_report", {"gaps": []})

        # Verify retrieval
        assert engine.retrieve_intermediate(session_id, "raw_extraction") == {
            "themes": ["auth"]
        }
        assert engine.retrieve_intermediate(session_id, "gap_report") == {"gaps": []}

        # Index stories for search
        stories = [
            {"item_id": "story-1", "content": "As a user I want to login"},
            {"item_id": "story-2", "content": "As an admin I want to manage users"},
        ]
        engine.store_for_search(session_id, stories)

        # Search should find results
        results = engine.search_similar("user login", top_k=5)
        assert len(results) >= 1

        # Log an action
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc),
            agent_name="extraction-agent",
            input_summary="Processing meeting transcript",
            output_summary="Extracted 5 themes",
            duration_ms=1500,
        )
        engine.log_action(session_id, entry)

        # Verify audit log
        log = engine.get_audit_log(session_id)
        assert len(log) == 1
        assert log[0].agent_name == "extraction-agent"
