"""Tests for gap detection pre-filtering behavior.

Verifies that the GapDetectionAgent:
- Excludes closed/archived/done/cancelled tickets from search
- Falls back to unfiltered search when filtering returns no results
- Works correctly when vector store doesn't support filtering
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from backlog_synthesizer.agents.gap_detection import GapDetectionAgent
from backlog_synthesizer.models.extraction import ExtractedItem
from backlog_synthesizer.tools.interfaces import SearchResult


def _make_item(text: str = "Add dark mode support") -> ExtractedItem:
    return ExtractedItem(
        item_type="feature_request",
        text=text,
        source_chunk_index=0,
        confidence=0.9,
    )


class TestGapDetectionPreFilter:
    """Tests for pre-filtering in gap detection."""

    async def test_filtered_query_called_when_supported(self):
        """When vector store supports query_similar_filtered, it's called with status filter."""
        mock_embedding = MagicMock()
        mock_embedding.generate_embedding = MagicMock(return_value=[0.1] * 384)

        mock_vector = MagicMock()
        mock_vector.query_similar_filtered = MagicMock(
            return_value=[
                SearchResult(item_id="TICKET-1", score=0.3, metadata={"status": "open"})
            ]
        )
        mock_vector.query_similar = MagicMock(return_value=[])

        agent = GapDetectionAgent(
            embedding_tool=mock_embedding,
            vector_search_tool=mock_vector,
        )

        report = await agent.analyze_gaps([_make_item()])

        # query_similar_filtered should be called with a where filter
        assert mock_vector.query_similar_filtered.called
        call_args = mock_vector.query_similar_filtered.call_args
        where_arg = call_args[1].get("where") or call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("where")
        assert where_arg is not None
        # Should exclude closed/archived/done/cancelled/canceled statuses
        assert "$and" in where_arg or "$ne" in str(where_arg)

    async def test_fallback_to_unfiltered_when_filtered_returns_empty(self):
        """Falls back to unfiltered search when filtered query returns no results."""
        mock_embedding = MagicMock()
        mock_embedding.generate_embedding = MagicMock(return_value=[0.1] * 384)

        mock_vector = MagicMock()
        # Filtered returns empty
        mock_vector.query_similar_filtered = MagicMock(return_value=[])
        # Unfiltered returns results
        mock_vector.query_similar = MagicMock(
            return_value=[
                SearchResult(item_id="CLOSED-1", score=0.95, metadata={"status": "closed"})
            ]
        )

        agent = GapDetectionAgent(
            embedding_tool=mock_embedding,
            vector_search_tool=mock_vector,
        )

        report = await agent.analyze_gaps([_make_item()])

        # Should have fallen back to unfiltered
        assert mock_vector.query_similar.called
        # Item should be classified based on the unfiltered result
        assert report.entries[0].classification == "duplicate"

    async def test_fallback_when_filter_not_supported(self):
        """Falls back to unfiltered search when vector store lacks query_similar_filtered."""
        mock_embedding = MagicMock()
        mock_embedding.generate_embedding = MagicMock(return_value=[0.1] * 384)

        mock_vector = MagicMock(spec=["query_similar", "store"])
        # No query_similar_filtered attribute
        mock_vector.query_similar = MagicMock(
            return_value=[
                SearchResult(item_id="TICKET-1", score=0.4, metadata={"status": "open"})
            ]
        )

        agent = GapDetectionAgent(
            embedding_tool=mock_embedding,
            vector_search_tool=mock_vector,
        )

        report = await agent.analyze_gaps([_make_item()])

        # Should work without filtering
        assert mock_vector.query_similar.called
        assert report.entries[0].classification == "new"

    async def test_excluded_statuses_are_correct(self):
        """Verify the set of excluded statuses."""
        assert "closed" in GapDetectionAgent.EXCLUDED_STATUSES
        assert "archived" in GapDetectionAgent.EXCLUDED_STATUSES
        assert "done" in GapDetectionAgent.EXCLUDED_STATUSES
        assert "cancelled" in GapDetectionAgent.EXCLUDED_STATUSES
        assert "canceled" in GapDetectionAgent.EXCLUDED_STATUSES
        # Active statuses should NOT be excluded
        assert "open" not in GapDetectionAgent.EXCLUDED_STATUSES
        assert "in_progress" not in GapDetectionAgent.EXCLUDED_STATUSES
        assert "backlog" not in GapDetectionAgent.EXCLUDED_STATUSES

    async def test_status_filter_structure(self):
        """Verify the built filter has correct ChromaDB structure."""
        agent = GapDetectionAgent(
            embedding_tool=MagicMock(),
            vector_search_tool=MagicMock(),
        )

        where_filter = agent._build_status_filter()

        assert where_filter is not None
        # Should be an $and clause with multiple $ne entries
        assert "$and" in where_filter
        ne_clauses = where_filter["$and"]
        assert len(ne_clauses) == len(GapDetectionAgent.EXCLUDED_STATUSES)
        for clause in ne_clauses:
            assert "status" in clause
            assert "$ne" in clause["status"]

    async def test_no_results_classifies_as_new(self):
        """When both filtered and unfiltered return empty, item is classified as NEW."""
        mock_embedding = MagicMock()
        mock_embedding.generate_embedding = MagicMock(return_value=[0.1] * 384)

        mock_vector = MagicMock()
        mock_vector.query_similar_filtered = MagicMock(return_value=[])
        mock_vector.query_similar = MagicMock(return_value=[])

        agent = GapDetectionAgent(
            embedding_tool=mock_embedding,
            vector_search_tool=mock_vector,
        )

        report = await agent.analyze_gaps([_make_item()])

        assert report.entries[0].classification == "new"
        assert report.entries[0].gap_type == "NEW"
        assert report.entries[0].confidence == 1.0
