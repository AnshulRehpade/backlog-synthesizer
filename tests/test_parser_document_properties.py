"""Property-based tests for ParserAgent document parsing.

Tests malformed document error structure and HTML stripping heading hierarchy preservation.
"""

import re

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backlog_synthesizer.agents.parser import ParserAgent, _HTMLToStructuredText
from backlog_synthesizer.models.extraction import DocumentError
from backlog_synthesizer.models.inputs import DocumentType, InputDocument
from backlog_synthesizer.tools.errors import ToolError


# --- Mock Tools ---


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


# --- Helper ---


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


# --- Strategies ---

# Strategy for generating non-empty filenames (printable, no whitespace-only)
filename_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        blacklist_characters="\x00/\\",
    ),
    min_size=1,
    max_size=50,
).filter(lambda s: len(s.strip()) > 0)

# Strategy for generating invalid UTF-8 byte sequences
# These bytes are guaranteed to NOT be valid UTF-8
invalid_utf8_bytes = st.binary(min_size=1, max_size=200).filter(
    lambda b: _is_invalid_utf8(b)
)


def _is_invalid_utf8(data: bytes) -> bool:
    """Return True if the bytes cannot be decoded as UTF-8."""
    try:
        data.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


# Strategy for heading levels (1-6)
heading_level = st.integers(min_value=1, max_value=6)

# Strategy for heading text content (no HTML special chars, non-empty)
heading_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Zs"),
        blacklist_characters="<>&\n\r\t",
    ),
    min_size=1,
    max_size=30,
).filter(lambda s: len(s.strip()) > 0)

# Strategy for body text content (no HTML special chars)
body_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Zs", "P"),
        blacklist_characters="<>&\n\r\t",
    ),
    min_size=0,
    max_size=50,
)

# Strategy for generating a list of headings with levels and text
heading_entry = st.tuples(heading_level, heading_text)
heading_list = st.lists(heading_entry, min_size=1, max_size=10)


# --- Property Tests ---


# Feature: backlog-synthesizer, Property 17: Malformed document error structure
class TestProperty17MalformedDocumentErrorStructure:
    """Verify that error returned from malformed documents contains non-empty filename and non-empty reason.

    **Validates: Requirements 1.5**
    """

    @given(filename=filename_strategy, content=invalid_utf8_bytes)
    @settings(max_examples=100)
    def test_invalid_utf8_text_produces_error_with_nonempty_fields(
        self, filename: str, content: bytes
    ) -> None:
        """
        For any invalid UTF-8 byte sequence fed as a text document,
        _ingest_document returns a DocumentError with non-empty filename and non-empty reason.

        **Validates: Requirements 1.5**
        """
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        doc = _make_doc(filename, content, DocumentType.TRANSCRIPT_TXT)
        text, error = parser._ingest_document(doc)

        assert text is None
        assert error is not None
        assert isinstance(error, DocumentError)
        assert len(error.filename) > 0, "Error filename must be non-empty"
        assert len(error.reason) > 0, "Error reason must be non-empty"

    @given(filename=filename_strategy)
    @settings(max_examples=100)
    def test_pdf_tool_failure_produces_error_with_nonempty_fields(
        self, filename: str
    ) -> None:
        """
        For any filename, when the PDF parsing tool fails, _ingest_document
        returns a DocumentError with non-empty filename and non-empty reason.

        **Validates: Requirements 1.5**
        """
        parser = ParserAgent(MockParsingTool(should_fail=True), MockLLMTool())
        doc = _make_doc(filename, b"%PDF-1.4 some content", DocumentType.TRANSCRIPT_PDF)
        text, error = parser._ingest_document(doc)

        assert text is None
        assert error is not None
        assert isinstance(error, DocumentError)
        assert len(error.filename) > 0, "Error filename must be non-empty"
        assert len(error.reason) > 0, "Error reason must be non-empty"

    @given(filename=filename_strategy)
    @settings(max_examples=100)
    def test_unsupported_type_produces_error_with_nonempty_fields(
        self, filename: str
    ) -> None:
        """
        For any filename with an unsupported document type (BACKLOG_JSON),
        _ingest_document returns a DocumentError with non-empty filename and non-empty reason.

        **Validates: Requirements 1.5**
        """
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        doc = _make_doc(filename, b'{"key": "value"}', DocumentType.BACKLOG_JSON)
        text, error = parser._ingest_document(doc)

        assert text is None
        assert error is not None
        assert isinstance(error, DocumentError)
        assert len(error.filename) > 0, "Error filename must be non-empty"
        assert len(error.reason) > 0, "Error reason must be non-empty"

    @given(filename=filename_strategy, content=invalid_utf8_bytes)
    @settings(max_examples=100)
    def test_invalid_utf8_html_produces_error_with_nonempty_fields(
        self, filename: str, content: bytes
    ) -> None:
        """
        For any invalid UTF-8 byte sequence fed as an HTML document,
        _ingest_document returns a DocumentError with non-empty filename and non-empty reason.

        **Validates: Requirements 1.5**
        """
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        doc = _make_doc(filename, content, DocumentType.ARCHITECTURE_HTML)
        text, error = parser._ingest_document(doc)

        assert text is None
        assert error is not None
        assert isinstance(error, DocumentError)
        assert len(error.filename) > 0, "Error filename must be non-empty"
        assert len(error.reason) > 0, "Error reason must be non-empty"


# Feature: backlog-synthesizer, Property 18: HTML stripping preserves heading hierarchy
class TestProperty18HTMLStrippingPreservesHeadingHierarchy:
    """Verify that HTML parsing strips all HTML tags and preserves heading order.

    **Validates: Requirements 1.3**
    """

    @given(headings=heading_list, bodies=st.lists(body_text, min_size=0, max_size=10))
    @settings(max_examples=100)
    def test_no_html_tags_in_output(
        self, headings: list[tuple[int, str]], bodies: list[str]
    ) -> None:
        """
        For any generated HTML with headings and body content,
        the parsed output contains no HTML tags (no '<' or '>' characters).

        **Validates: Requirements 1.3**
        """
        html = _build_html(headings, bodies)
        parser = _HTMLToStructuredText()
        parser.feed(html)
        parser.close()
        result = parser.get_result()

        assert "<" not in result, f"Output contains '<': {result!r}"
        assert ">" not in result, f"Output contains '>': {result!r}"

    @given(headings=heading_list, bodies=st.lists(body_text, min_size=0, max_size=10))
    @settings(max_examples=100)
    def test_heading_order_preserved(
        self, headings: list[tuple[int, str]], bodies: list[str]
    ) -> None:
        """
        For any generated HTML with headings, the order of headings in the output
        matches the order in the source HTML (heading A before heading B in source
        implies heading A before heading B in output).

        **Validates: Requirements 1.3**
        """
        html = _build_html(headings, bodies)
        parser = _HTMLToStructuredText()
        parser.feed(html)
        parser.close()
        result = parser.get_result()

        # Extract headings from the output (lines starting with one or more '#')
        output_headings: list[str] = []
        for line in result.split("\n"):
            stripped = line.strip()
            if stripped and stripped.startswith("#"):
                # Extract the heading text after the '# ' prefix
                match = re.match(r"^#+\s+(.*)", stripped)
                if match:
                    output_headings.append(match.group(1).strip())

        # Build expected heading texts in order (only those with non-empty stripped text)
        expected_headings = [text.strip() for _level, text in headings if text.strip()]

        # Verify each expected heading appears in order in the output
        output_idx = 0
        for expected in expected_headings:
            found = False
            while output_idx < len(output_headings):
                if output_headings[output_idx] == expected:
                    found = True
                    output_idx += 1
                    break
                output_idx += 1
            assert found, (
                f"Heading '{expected}' not found in correct order in output.\n"
                f"Expected order: {expected_headings}\n"
                f"Output headings: {output_headings}"
            )

    @given(headings=heading_list)
    @settings(max_examples=100)
    def test_heading_level_prefix_matches_source(
        self, headings: list[tuple[int, str]]
    ) -> None:
        """
        For any heading at level N in the source HTML, the output contains
        the heading text prefixed with exactly N '#' characters.

        **Validates: Requirements 1.3**
        """
        html = _build_html(headings, [])
        parser = _HTMLToStructuredText()
        parser.feed(html)
        parser.close()
        result = parser.get_result()

        for level, text in headings:
            stripped_text = text.strip()
            if not stripped_text:
                continue
            expected_prefix = "#" * level
            expected_line = f"{expected_prefix} {stripped_text}"
            assert expected_line in result, (
                f"Expected '{expected_line}' in output but not found.\n"
                f"Output: {result!r}"
            )


def _build_html(
    headings: list[tuple[int, str]], bodies: list[str]
) -> str:
    """Build an HTML string from a list of (level, text) heading tuples and body texts.

    Interleaves body paragraphs between headings where available.
    """
    parts: list[str] = ["<html><body>"]
    for i, (level, text) in enumerate(headings):
        parts.append(f"<h{level}>{text}</h{level}>")
        # Add a body paragraph after each heading if available
        if i < len(bodies) and bodies[i]:
            parts.append(f"<p>{bodies[i]}</p>")
    # Add remaining body paragraphs
    for body in bodies[len(headings):]:
        if body:
            parts.append(f"<p>{body}</p>")
    parts.append("</body></html>")
    return "".join(parts)
