"""Unit tests for ParserAgent document ingestion methods."""

import pytest

from backlog_synthesizer.agents.parser import ParserAgent, _HTMLToStructuredText
from backlog_synthesizer.models.extraction import DocumentError
from backlog_synthesizer.models.inputs import DocumentType, InputDocument
from backlog_synthesizer.tools.errors import ToolError, TransientToolError


class MockParsingTool:
    """Mock DocumentParsingTool for testing."""

    def __init__(self, pdf_text: str = "Parsed PDF content.", should_fail: bool = False):
        self._pdf_text = pdf_text
        self._should_fail = should_fail

    def pdf_to_text(self, content: bytes) -> str:
        if self._should_fail:
            raise ToolError("PDF parsing failed: corrupted file")
        return self._pdf_text

    def chunk_text(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        return []


class MockLLMTool:
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return ""


def _make_doc(
    filename: str,
    content: bytes,
    doc_type: DocumentType,
) -> InputDocument:
    return InputDocument(
        filename=filename,
        document_type=doc_type,
        content=content,
        size_bytes=len(content),
    )


# ---------- Text/Markdown Ingestion Tests ----------


class TestIngestText:
    """Tests for _ingest_text handling .txt and .md files."""

    def test_plain_text_decoded(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        doc = _make_doc("notes.txt", b"Hello, world!", DocumentType.TRANSCRIPT_TXT)
        result = parser._ingest_text(doc)
        assert result == "Hello, world!"

    def test_markdown_content_preserved(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        content = b"# Meeting Notes\n\n- Decision: Use Python\n- Action: Setup CI"
        doc = _make_doc("meeting.md", content, DocumentType.TRANSCRIPT_MD)
        result = parser._ingest_text(doc)
        assert "# Meeting Notes" in result
        assert "- Decision: Use Python" in result

    def test_utf8_multibyte_characters(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        content = "Ünîcödé tëxt with émojis 🎉".encode("utf-8")
        doc = _make_doc("unicode.txt", content, DocumentType.TRANSCRIPT_TXT)
        result = parser._ingest_text(doc)
        assert "Ünîcödé" in result
        assert "🎉" in result

    def test_invalid_utf8_raises_unicode_error(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        # Invalid UTF-8 byte sequence
        doc = _make_doc("bad.txt", b"\xff\xfe invalid", DocumentType.TRANSCRIPT_TXT)
        with pytest.raises(UnicodeDecodeError):
            parser._ingest_text(doc)


# ---------- PDF Ingestion Tests ----------


class TestIngestPdf:
    """Tests for _ingest_pdf using DocumentParsingTool."""

    def test_pdf_calls_parsing_tool(self) -> None:
        mock_tool = MockParsingTool(pdf_text="Extracted from PDF.")
        parser = ParserAgent(mock_tool, MockLLMTool())
        doc = _make_doc("report.pdf", b"%PDF-1.4 fake content", DocumentType.TRANSCRIPT_PDF)
        result = parser._ingest_pdf(doc)
        assert result == "Extracted from PDF."

    def test_pdf_tool_error_propagates(self) -> None:
        mock_tool = MockParsingTool(should_fail=True)
        parser = ParserAgent(mock_tool, MockLLMTool())
        doc = _make_doc("broken.pdf", b"%PDF-1.4 corrupted", DocumentType.TRANSCRIPT_PDF)
        with pytest.raises(ToolError):
            parser._ingest_pdf(doc)


# ---------- HTML Ingestion Tests ----------


class TestIngestHtml:
    """Tests for _ingest_html with markup stripping and heading preservation."""

    def test_simple_html_stripped(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        html = b"<html><body><p>Hello world</p></body></html>"
        doc = _make_doc("wiki.html", html, DocumentType.ARCHITECTURE_HTML)
        result = parser._ingest_html(doc)
        assert "<" not in result
        assert "Hello world" in result

    def test_heading_hierarchy_preserved(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        html = b"<h1>Title</h1><h2>Section</h2><p>Content</p><h3>Subsection</h3>"
        doc = _make_doc("arch.html", html, DocumentType.ARCHITECTURE_HTML)
        result = parser._ingest_html(doc)
        assert "# Title" in result
        assert "## Section" in result
        assert "### Subsection" in result
        # Verify ordering
        title_pos = result.index("# Title")
        section_pos = result.index("## Section")
        subsection_pos = result.index("### Subsection")
        assert title_pos < section_pos < subsection_pos

    def test_all_heading_levels(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        html = (
            b"<h1>H1</h1><h2>H2</h2><h3>H3</h3>"
            b"<h4>H4</h4><h5>H5</h5><h6>H6</h6>"
        )
        doc = _make_doc("headings.html", html, DocumentType.ARCHITECTURE_HTML)
        result = parser._ingest_html(doc)
        assert "# H1" in result
        assert "## H2" in result
        assert "### H3" in result
        assert "#### H4" in result
        assert "##### H5" in result
        assert "###### H6" in result

    def test_no_html_tags_in_output(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        html = (
            b"<html><head><title>Page</title></head>"
            b"<body><div class='content'><h1>Title</h1>"
            b"<p>Some <strong>bold</strong> and <em>italic</em> text.</p>"
            b"<ul><li>Item 1</li><li>Item 2</li></ul></div></body></html>"
        )
        doc = _make_doc("page.html", html, DocumentType.ARCHITECTURE_HTML)
        result = parser._ingest_html(doc)
        assert "<" not in result
        assert ">" not in result

    def test_text_content_preserved_from_html(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        html = b"<p>First paragraph.</p><p>Second paragraph.</p>"
        doc = _make_doc("doc.html", html, DocumentType.ARCHITECTURE_HTML)
        result = parser._ingest_html(doc)
        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_empty_html_returns_empty_string(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        doc = _make_doc("empty.html", b"", DocumentType.ARCHITECTURE_HTML)
        result = parser._ingest_html(doc)
        assert result == ""


# ---------- Document Routing (_ingest_document) Tests ----------


class TestIngestDocument:
    """Tests for the _ingest_document routing method."""

    def test_routes_txt_correctly(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        doc = _make_doc("file.txt", b"text content", DocumentType.TRANSCRIPT_TXT)
        text, error = parser._ingest_document(doc)
        assert text == "text content"
        assert error is None

    def test_routes_md_correctly(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        doc = _make_doc("file.md", b"# Markdown", DocumentType.TRANSCRIPT_MD)
        text, error = parser._ingest_document(doc)
        assert text == "# Markdown"
        assert error is None

    def test_routes_pdf_correctly(self) -> None:
        mock_tool = MockParsingTool(pdf_text="PDF text output")
        parser = ParserAgent(mock_tool, MockLLMTool())
        doc = _make_doc("file.pdf", b"%PDF-1.4", DocumentType.TRANSCRIPT_PDF)
        text, error = parser._ingest_document(doc)
        assert text == "PDF text output"
        assert error is None

    def test_routes_html_correctly(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        doc = _make_doc("wiki.html", b"<h1>Title</h1><p>Body</p>", DocumentType.ARCHITECTURE_HTML)
        text, error = parser._ingest_document(doc)
        assert text is not None
        assert "# Title" in text
        assert "Body" in text
        assert error is None

    def test_unicode_error_returns_document_error(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        doc = _make_doc("bad.txt", b"\xff\xfe\x00\x01", DocumentType.TRANSCRIPT_TXT)
        text, error = parser._ingest_document(doc)
        assert text is None
        assert error is not None
        assert error.filename == "bad.txt"
        assert error.reason != ""
        assert error.byte_offset is not None

    def test_tool_error_returns_document_error(self) -> None:
        mock_tool = MockParsingTool(should_fail=True)
        parser = ParserAgent(mock_tool, MockLLMTool())
        doc = _make_doc("broken.pdf", b"%PDF corrupted", DocumentType.TRANSCRIPT_PDF)
        text, error = parser._ingest_document(doc)
        assert text is None
        assert error is not None
        assert error.filename == "broken.pdf"
        assert "tool error" in error.reason.lower() or "parsing" in error.reason.lower()

    def test_unsupported_type_returns_document_error(self) -> None:
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        doc = _make_doc("data.json", b'{"key": "value"}', DocumentType.BACKLOG_JSON)
        text, error = parser._ingest_document(doc)
        assert text is None
        assert error is not None
        assert error.filename == "data.json"
        assert "unsupported" in error.reason.lower()

    def test_document_error_has_nonempty_filename_and_reason(self) -> None:
        """Validates Property 17: error contains non-empty filename and reason."""
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        doc = _make_doc("corrupt.txt", b"\x80\x81\x82", DocumentType.TRANSCRIPT_TXT)
        text, error = parser._ingest_document(doc)
        assert text is None
        assert error is not None
        assert len(error.filename) > 0
        assert len(error.reason) > 0


# ---------- HTML Parser Internal Tests ----------


class TestHTMLToStructuredText:
    """Tests for the internal HTML parser class."""

    def test_nested_headings_with_inline_tags(self) -> None:
        parser = _HTMLToStructuredText()
        parser.feed("<h2>Section with <em>emphasis</em></h2>")
        parser.close()
        result = parser.get_result()
        assert "## Section with emphasis" in result
        assert "<em>" not in result

    def test_multiple_paragraphs_separated(self) -> None:
        parser = _HTMLToStructuredText()
        parser.feed("<p>First.</p><p>Second.</p>")
        parser.close()
        result = parser.get_result()
        assert "First." in result
        assert "Second." in result
        # They should be separate (not merged into one line)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) >= 2
