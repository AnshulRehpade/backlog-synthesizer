"""Unit tests for ChromaVectorSearchTool with mocked chromadb."""

from unittest.mock import MagicMock, patch

import pytest

from backlog_synthesizer.tools.errors import PermanentToolError
from backlog_synthesizer.tools.interfaces import SearchResult


@pytest.fixture
def mock_chromadb():
    """Patch chromadb import inside vector_search module."""
    with patch.dict("sys.modules", {"chromadb": MagicMock()}):
        import sys

        mock_chroma = sys.modules["chromadb"]
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.EphemeralClient.return_value = mock_client
        yield mock_chroma, mock_collection


@pytest.fixture
def tool(mock_chromadb):
    """Create a ChromaVectorSearchTool with mocked chromadb."""
    from backlog_synthesizer.tools.vector_search import ChromaVectorSearchTool

    t = ChromaVectorSearchTool(collection_name="test_col")
    _, mock_collection = mock_chromadb
    t._collection = mock_collection
    return t


class TestChromaVectorSearchTool:
    def test_store_calls_upsert(self, tool, mock_chromadb):
        _, collection = mock_chromadb
        tool.store("item-1", [0.1, 0.2, 0.3], {"key": "value"})
        collection.upsert.assert_called_once()

    def test_store_empty_item_id_raises(self, tool):
        with pytest.raises(PermanentToolError, match="non-empty string"):
            tool.store("", [0.1, 0.2], {"key": "val"})

    def test_store_empty_embedding_raises(self, tool):
        with pytest.raises(PermanentToolError, match="non-empty list"):
            tool.store("item-1", [], {"key": "val"})

    def test_query_similar_returns_results(self, tool, mock_chromadb):
        _, collection = mock_chromadb
        collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"k": "v1"}, {"k": "v2"}]],
        }

        results = tool.query_similar([0.1, 0.2], top_k=2)

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        # score = 1 - distance, so 0.1 distance = 0.9 score (highest first)
        assert results[0].score == pytest.approx(0.9)
        assert results[1].score == pytest.approx(0.7)

    def test_query_similar_empty_embedding_raises(self, tool):
        with pytest.raises(PermanentToolError, match="non-empty list"):
            tool.query_similar([], top_k=5)

    def test_query_similar_filtered_passes_where(self, tool, mock_chromadb):
        _, collection = mock_chromadb
        collection.query.return_value = {
            "ids": [["id1"]],
            "distances": [[0.2]],
            "metadatas": [[{"status": "open"}]],
        }
        where_clause = {"status": {"$ne": "closed"}}

        tool.query_similar_filtered([0.1], top_k=1, where=where_clause)

        call_kwargs = collection.query.call_args[1]
        assert call_kwargs["where"] == where_clause

    def test_sanitize_metadata_converts_types(self, tool):
        metadata = {
            "str_key": "value",
            "int_key": 42,
            "float_key": 3.14,
            "bool_key": True,
            "none_key": None,
            "list_key": [1, 2, 3],
        }

        result = tool._sanitize_metadata(metadata)

        assert result["str_key"] == "value"
        assert result["int_key"] == 42
        assert result["float_key"] == 3.14
        assert result["bool_key"] is True
        assert "none_key" not in result
        assert result["list_key"] == "[1, 2, 3]"
