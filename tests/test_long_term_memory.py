"""Tests for Long-Term Memory implementation.

Uses mock implementations of EmbeddingTool and VectorSearchTool to verify
storage, search, and retention policy behavior.
"""

from datetime import datetime, timedelta, timezone

import pytest

from backlog_synthesizer.memory.long_term import LongTermMemory
from backlog_synthesizer.tools.interfaces import SearchResult


class MockEmbeddingTool:
    """Mock EmbeddingTool that returns a deterministic embedding based on text hash."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_text: str | None = None

    def generate_embedding(self, text: str) -> list[float]:
        self.call_count += 1
        self.last_text = text
        # Produce a simple deterministic embedding from the text
        hash_val = hash(text) % 1000
        return [float(hash_val), float(hash_val + 1), float(hash_val + 2)]


class MockVectorSearchTool:
    """Mock VectorSearchTool that stores items in-memory for testing."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[list[float], dict]] = {}
        self.store_calls: list[tuple[str, list[float], dict]] = []
        self.query_calls: list[tuple[list[float], int]] = []

    def store(self, item_id: str, embedding: list[float], metadata: dict) -> None:
        self._store[item_id] = (embedding, metadata)
        self.store_calls.append((item_id, embedding, metadata))

    def query_similar(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        self.query_calls.append((embedding, top_k))
        # Return all stored items as results (simple mock behavior)
        results = []
        for item_id, (stored_embedding, metadata) in self._store.items():
            results.append(
                SearchResult(item_id=item_id, score=0.9, metadata=metadata)
            )
        return results[:top_k]


class TestLongTermMemoryInit:
    """Tests for LongTermMemory initialization."""

    def test_default_retention_days(self) -> None:
        memory = LongTermMemory(MockEmbeddingTool(), MockVectorSearchTool())
        assert memory.retention_days == 30

    def test_custom_retention_days(self) -> None:
        memory = LongTermMemory(
            MockEmbeddingTool(), MockVectorSearchTool(), retention_days=60
        )
        assert memory.retention_days == 60


class TestStoreItem:
    """Tests for storing items in long-term memory."""

    def test_store_generates_embedding(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool)

        memory.store_item("item-1", "User wants a login feature")

        assert embedding_tool.call_count == 1
        assert embedding_tool.last_text == "User wants a login feature"

    def test_store_passes_embedding_to_vector_store(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool)

        memory.store_item("item-1", "User wants a login feature")

        assert len(vector_tool.store_calls) == 1
        stored_id, stored_embedding, _ = vector_tool.store_calls[0]
        assert stored_id == "item-1"
        expected_embedding = embedding_tool.generate_embedding(
            "User wants a login feature"
        )
        assert stored_embedding == expected_embedding

    def test_store_includes_stored_at_timestamp(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool)

        memory.store_item("item-1", "Some content")

        _, _, metadata = vector_tool.store_calls[0]
        assert "stored_at" in metadata
        # Verify it's a valid ISO 8601 timestamp
        stored_at = datetime.fromisoformat(metadata["stored_at"])
        assert stored_at.tzinfo is not None

    def test_store_includes_content_in_metadata(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool)

        memory.store_item("item-1", "Important content")

        _, _, metadata = vector_tool.store_calls[0]
        assert metadata["content"] == "Important content"

    def test_store_merges_user_metadata(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool)

        memory.store_item(
            "item-1",
            "Content",
            metadata={"session_id": "sess-001", "item_type": "feature_request"},
        )

        _, _, metadata = vector_tool.store_calls[0]
        assert metadata["session_id"] == "sess-001"
        assert metadata["item_type"] == "feature_request"
        assert "stored_at" in metadata
        assert "content" in metadata

    def test_store_with_no_metadata(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool)

        memory.store_item("item-1", "Content")

        _, _, metadata = vector_tool.store_calls[0]
        assert "stored_at" in metadata
        assert "content" in metadata


class TestSearchSimilar:
    """Tests for semantic search in long-term memory."""

    def test_search_generates_query_embedding(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool)

        memory.search_similar("login feature")

        assert embedding_tool.call_count == 1
        assert embedding_tool.last_text == "login feature"

    def test_search_queries_vector_store_with_embedding(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool)

        memory.search_similar("login feature", top_k=5)

        assert len(vector_tool.query_calls) == 1
        queried_embedding, queried_top_k = vector_tool.query_calls[0]
        expected_embedding = embedding_tool.generate_embedding("login feature")
        assert queried_embedding == expected_embedding
        assert queried_top_k == 5

    def test_search_default_top_k(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool)

        memory.search_similar("query text")

        _, queried_top_k = vector_tool.query_calls[0]
        assert queried_top_k == 10

    def test_search_returns_results(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool)

        # Store some items first
        memory.store_item("item-1", "Login feature")
        memory.store_item("item-2", "Auth system")

        results = memory.search_similar("authentication")

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_empty_store_returns_empty(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool)

        results = memory.search_similar("anything")

        assert results == []


class TestPurgeExpired:
    """Tests for the 30-day retention policy purge mechanism."""

    def test_purge_identifies_expired_items(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool, retention_days=30)

        # Manually insert an item with an old timestamp
        old_timestamp = (
            datetime.now(timezone.utc) - timedelta(days=31)
        ).isoformat()
        vector_tool._store["old-item"] = (
            [1.0, 2.0, 3.0],
            {"stored_at": old_timestamp, "content": "old content"},
        )

        expired = memory.purge_expired()

        assert "old-item" in expired

    def test_purge_keeps_recent_items(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool, retention_days=30)

        # Manually insert a recent item
        recent_timestamp = (
            datetime.now(timezone.utc) - timedelta(days=5)
        ).isoformat()
        vector_tool._store["recent-item"] = (
            [1.0, 2.0, 3.0],
            {"stored_at": recent_timestamp, "content": "recent content"},
        )

        expired = memory.purge_expired()

        assert "recent-item" not in expired

    def test_purge_with_custom_reference_time(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool, retention_days=30)

        # Item stored 20 days ago
        stored_time = datetime.now(timezone.utc) - timedelta(days=20)
        vector_tool._store["item-1"] = (
            [1.0, 2.0, 3.0],
            {"stored_at": stored_time.isoformat(), "content": "content"},
        )

        # Using reference time 15 days in the future makes it 35 days old
        future_ref = datetime.now(timezone.utc) + timedelta(days=15)
        expired = memory.purge_expired(reference_time=future_ref)

        assert "item-1" in expired

    def test_purge_skips_items_without_stored_at(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool, retention_days=30)

        # Item without stored_at metadata
        vector_tool._store["no-timestamp"] = (
            [1.0, 2.0, 3.0],
            {"content": "no timestamp"},
        )

        expired = memory.purge_expired()

        assert "no-timestamp" not in expired

    def test_purge_mixed_items(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool, retention_days=30)

        now = datetime.now(timezone.utc)

        # Old item (expired)
        vector_tool._store["old"] = (
            [1.0],
            {"stored_at": (now - timedelta(days=45)).isoformat(), "content": "old"},
        )
        # Recent item (not expired)
        vector_tool._store["recent"] = (
            [2.0],
            {"stored_at": (now - timedelta(days=10)).isoformat(), "content": "new"},
        )
        # Boundary item (exactly 30 days, not expired since > is required)
        vector_tool._store["boundary"] = (
            [3.0],
            {"stored_at": (now - timedelta(days=30)).isoformat(), "content": "edge"},
        )

        expired = memory.purge_expired(reference_time=now)

        assert "old" in expired
        assert "recent" not in expired
        assert "boundary" not in expired

    def test_purge_returns_empty_for_empty_store(self) -> None:
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool)

        expired = memory.purge_expired()

        assert expired == []

    def test_store_item_then_purge_within_retention(self) -> None:
        """Items stored through store_item should not be expired immediately."""
        embedding_tool = MockEmbeddingTool()
        vector_tool = MockVectorSearchTool()
        memory = LongTermMemory(embedding_tool, vector_tool, retention_days=30)

        memory.store_item("fresh-item", "Just stored this")

        expired = memory.purge_expired()

        assert "fresh-item" not in expired
