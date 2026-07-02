"""Unit tests for the GapDetectionAgent class.

Tests the agent's analyze_gaps method with mocked tool implementations
to verify integration of embedding, search, and classification.
"""

import asyncio

import pytest

from backlog_synthesizer.agents.gap_detection import GapDetectionAgent
from backlog_synthesizer.models.extraction import ExtractedItem
from backlog_synthesizer.tools.interfaces import SearchResult


# --- Mock Tool Implementations ---


class MockEmbeddingTool:
    """Mock embedding tool that returns a fixed embedding vector."""

    def __init__(self, embedding: list[float] | None = None):
        self._embedding = embedding or [0.1, 0.2, 0.3]

    def generate_embedding(self, text: str) -> list[float]:
        return self._embedding


class MockVectorSearchTool:
    """Mock vector search tool with configurable results."""

    def __init__(self, results: list[SearchResult] | None = None):
        self._results = results if results is not None else []

    def query_similar(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        return self._results

    def store(self, item_id: str, embedding: list[float], metadata: dict) -> None:
        pass


class MockLLMGenerationTool:
    """Mock LLM tool with configurable response."""

    def __init__(self, response: str = "NO CONTRADICTION"):
        self._response = response

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self._response


class SlowEmbeddingTool:
    """Embedding tool that sleeps to simulate timeout."""

    def __init__(self, delay: float):
        self._delay = delay

    def generate_embedding(self, text: str) -> list[float]:
        import time
        time.sleep(self._delay)
        return [0.1, 0.2, 0.3]


class FailingEmbeddingTool:
    """Embedding tool that raises an exception."""

    def generate_embedding(self, text: str) -> list[float]:
        raise RuntimeError("Embedding service unavailable")


# --- Helper ---


def make_item(text: str = "Test feature request", item_type: str = "feature_request") -> ExtractedItem:
    return ExtractedItem(
        item_type=item_type,
        text=text,
        source_chunk_index=0,
        confidence=0.9,
    )


# --- Tests ---


class TestGapDetectionAgentEmptyBacklog:
    """Test behavior when vector search returns no results (empty backlog)."""

    @pytest.mark.asyncio
    async def test_empty_backlog_classifies_all_as_new(self) -> None:
        agent = GapDetectionAgent(
            embedding_tool=MockEmbeddingTool(),
            vector_search_tool=MockVectorSearchTool(results=[]),
        )
        items = [make_item("Feature A"), make_item("Feature B")]
        report = await agent.analyze_gaps(items)

        assert report.total_new == 2
        assert report.total_duplicates == 0
        assert report.total_conflicts == 0
        assert report.total_unprocessed == 0
        for entry in report.entries:
            assert entry.classification == "new"
            assert entry.confidence == 1.0


class TestGapDetectionAgentDuplicate:
    """Test duplicate classification when similarity >= 0.85."""

    @pytest.mark.asyncio
    async def test_high_similarity_classified_as_duplicate(self) -> None:
        results = [SearchResult(item_id="TICKET-123", score=0.92)]
        agent = GapDetectionAgent(
            embedding_tool=MockEmbeddingTool(),
            vector_search_tool=MockVectorSearchTool(results=results),
        )
        items = [make_item("Add login button")]
        report = await agent.analyze_gaps(items)

        assert report.total_duplicates == 1
        assert report.entries[0].classification == "duplicate"
        assert report.entries[0].duplicate_info is not None
        assert report.entries[0].duplicate_info.matching_ticket_id == "TICKET-123"


class TestGapDetectionAgentConflict:
    """Test conflict classification when similarity is 0.50-0.85 with contradiction."""

    @pytest.mark.asyncio
    async def test_mid_similarity_with_contradiction_classified_as_conflict(self) -> None:
        results = [SearchResult(item_id="TICKET-456", score=0.70, metadata={"text": "Use REST API"})]
        agent = GapDetectionAgent(
            embedding_tool=MockEmbeddingTool(),
            vector_search_tool=MockVectorSearchTool(results=results),
            llm_tool=MockLLMGenerationTool(response="CONTRADICTION: REST vs GraphQL approach"),
        )
        items = [make_item("Use GraphQL API")]
        report = await agent.analyze_gaps(items)

        assert report.total_conflicts == 1
        assert report.entries[0].classification == "conflict"
        assert report.entries[0].conflict_info is not None
        assert "REST vs GraphQL" in report.entries[0].conflict_info.contradiction_description

    @pytest.mark.asyncio
    async def test_mid_similarity_without_contradiction_classified_as_new(self) -> None:
        results = [SearchResult(item_id="TICKET-789", score=0.65, metadata={"text": "Add dashboard"})]
        agent = GapDetectionAgent(
            embedding_tool=MockEmbeddingTool(),
            vector_search_tool=MockVectorSearchTool(results=results),
            llm_tool=MockLLMGenerationTool(response="NO CONTRADICTION"),
        )
        items = [make_item("Improve dashboard layout")]
        report = await agent.analyze_gaps(items)

        assert report.total_new == 1
        assert report.entries[0].classification == "new"


class TestGapDetectionAgentTimeout:
    """Test timeout handling for per-request processing."""

    @pytest.mark.asyncio
    async def test_timeout_marks_item_as_unprocessed(self) -> None:
        agent = GapDetectionAgent(
            embedding_tool=SlowEmbeddingTool(delay=5.0),
            vector_search_tool=MockVectorSearchTool(),
        )
        items = [make_item("Slow item")]
        report = await agent.analyze_gaps(items, timeout=0.1)

        assert report.total_unprocessed == 1
        assert report.entries[0].classification == "unprocessed"
        assert "timed out" in report.entries[0].error_reason


class TestGapDetectionAgentErrors:
    """Test error handling when tools raise exceptions."""

    @pytest.mark.asyncio
    async def test_exception_marks_item_as_unprocessed(self) -> None:
        agent = GapDetectionAgent(
            embedding_tool=FailingEmbeddingTool(),
            vector_search_tool=MockVectorSearchTool(),
        )
        items = [make_item("Error item")]
        report = await agent.analyze_gaps(items)

        assert report.total_unprocessed == 1
        assert report.entries[0].classification == "unprocessed"
        assert "unavailable" in report.entries[0].error_reason


class TestGapDetectionAgentReportCounts:
    """Test that report counts are accurate across mixed classifications."""

    @pytest.mark.asyncio
    async def test_mixed_classifications_produce_correct_counts(self) -> None:
        # Item 1: will be duplicate (score 0.90)
        # Item 2: will be new (empty results)
        # We need different results per item, so let's use a stateful mock

        class StatefulVectorSearch:
            def __init__(self):
                self._call_count = 0
                self._results_list = [
                    [SearchResult(item_id="T-1", score=0.90)],  # duplicate
                    [],  # new (empty backlog)
                ]

            def query_similar(self, embedding: list[float], top_k: int) -> list[SearchResult]:
                idx = self._call_count
                self._call_count += 1
                if idx < len(self._results_list):
                    return self._results_list[idx]
                return []

            def store(self, item_id: str, embedding: list[float], metadata: dict) -> None:
                pass

        agent = GapDetectionAgent(
            embedding_tool=MockEmbeddingTool(),
            vector_search_tool=StatefulVectorSearch(),
        )
        items = [make_item("Duplicate item"), make_item("New item")]
        report = await agent.analyze_gaps(items)

        assert report.total_duplicates == 1
        assert report.total_new == 1
        assert report.total_conflicts == 0
        assert report.total_unprocessed == 0
        assert len(report.entries) == 2
