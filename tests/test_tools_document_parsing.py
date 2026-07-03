"""Unit tests for PyMuPDFDocumentParser with mocked fitz (PyMuPDF)."""

from unittest.mock import MagicMock, patch

import pytest

from backlog_synthesizer.tools.errors import PermanentToolError


class TestPyMuPDFDocumentParser:
    @patch("backlog_synthesizer.tools.document_parsing.fitz")
    def test_pdf_to_text_extracts_pages(self, mock_fitz):
        from backlog_synthesizer.tools.document_parsing import PyMuPDFDocumentParser

        page1 = MagicMock()
        page1.get_text.return_value = "Page one content"
        page2 = MagicMock()
        page2.get_text.return_value = "Page two content"
        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([page1, page2])
        mock_fitz.open.return_value = mock_doc

        parser = PyMuPDFDocumentParser()
        result = parser.pdf_to_text(b"fake pdf bytes")

        assert "Page one content" in result
        assert "Page two content" in result
        mock_doc.close.assert_called_once()

    @patch("backlog_synthesizer.tools.document_parsing.fitz")
    def test_empty_pdf_raises_error(self, mock_fitz):
        from backlog_synthesizer.tools.document_parsing import PyMuPDFDocumentParser

        parser = PyMuPDFDocumentParser()

        with pytest.raises(PermanentToolError, match="empty PDF"):
            parser.pdf_to_text(b"")

    @patch("backlog_synthesizer.tools.document_parsing.fitz")
    def test_exception_raises_tool_error(self, mock_fitz):
        from backlog_synthesizer.tools.document_parsing import PyMuPDFDocumentParser

        mock_fitz.open.side_effect = RuntimeError("corrupted file")
        parser = PyMuPDFDocumentParser()

        with pytest.raises(PermanentToolError, match="Failed to open PDF"):
            parser.pdf_to_text(b"corrupted data")

    def test_chunk_text_removed_from_protocol(self):
        """chunk_text was removed — chunking is handled by ParserAgent internally."""
        from backlog_synthesizer.tools.document_parsing import PyMuPDFDocumentParser

        parser = PyMuPDFDocumentParser()
        assert not hasattr(parser, "chunk_text")
