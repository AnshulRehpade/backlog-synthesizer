"""Unit tests for SentenceTransformerEmbeddingTool with mocked sentence_transformers."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backlog_synthesizer.tools.errors import PermanentToolError, TransientToolError


class TestSentenceTransformerEmbeddingTool:
    @patch("backlog_synthesizer.tools.embedding.SentenceTransformerEmbeddingTool._load_model")
    def test_generate_embedding_returns_floats(self, mock_load):
        from backlog_synthesizer.tools.embedding import SentenceTransformerEmbeddingTool

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_load.return_value = mock_model

        tool = SentenceTransformerEmbeddingTool(model_name="test-model")
        result = tool.generate_embedding("hello world")

        assert result == pytest.approx([0.1, 0.2, 0.3])
        assert all(isinstance(v, float) for v in result)

    @patch("backlog_synthesizer.tools.embedding.SentenceTransformerEmbeddingTool._load_model")
    def test_import_error_raises_permanent(self, mock_load):
        """Simulate ImportError during model loading."""
        from backlog_synthesizer.tools.embedding import SentenceTransformerEmbeddingTool

        mock_load.side_effect = PermanentToolError(
            "sentence-transformers library is not installed. "
            "Install it with: pip install sentence-transformers"
        )

        with pytest.raises(PermanentToolError, match="not installed"):
            SentenceTransformerEmbeddingTool(model_name="test-model")

    @patch("backlog_synthesizer.tools.embedding.SentenceTransformerEmbeddingTool._load_model")
    def test_runtime_error_raises_transient(self, mock_load):
        from backlog_synthesizer.tools.embedding import SentenceTransformerEmbeddingTool

        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("CUDA OOM")
        mock_load.return_value = mock_model

        tool = SentenceTransformerEmbeddingTool(model_name="test-model")

        with pytest.raises(TransientToolError, match="inference failed"):
            tool.generate_embedding("hello")
