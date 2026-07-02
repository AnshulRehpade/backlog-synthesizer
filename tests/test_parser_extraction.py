"""Unit tests for ParserAgent information extraction via LLM (Task 4.5).

Tests verify extraction of decisions, pain points, and feature requests from
meeting transcripts, and technical constraints/decisions from architecture documents.
"""

import json

import pytest

from backlog_synthesizer.agents.parser import ParserAgent
from backlog_synthesizer.models.extraction import ExtractedItem, ExtractionResult
from backlog_synthesizer.models.inputs import DocumentType, InputDocument
from backlog_synthesizer.tools.errors import ToolError, TransientToolError


class MockParsingTool:
    """Mock DocumentParsingTool for testing."""

    def __init__(self, pdf_text: str = "Parsed PDF content."):
        self._pdf_text = pdf_text

    def pdf_to_text(self, content: bytes) -> str:
        return self._pdf_text

    def chunk_text(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        return []


class MockLLMTool:
    """Mock LLM tool that returns configurable responses per call."""

    def __init__(self, responses: list[str] | None = None, error: Exception | None = None):
        self._responses = responses or []
        self._call_index = 0
        self._error = error
        self.calls: list[dict] = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        if self._error:
            raise self._error
        if self._call_index < len(self._responses):
            response = self._responses[self._call_index]
            self._call_index += 1
            return response
        return "[]"


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


# ---------- parse_documents Integration Tests ----------


class TestParseDocuments:
    """Tests for the parse_documents method orchestrating ingestion and extraction."""

    @pytest.mark.asyncio
    async def test_transcript_extraction_returns_items(self) -> None:
        """Validates: Requirements 3.1, 3.2, 3.3"""
        llm_response = json.dumps([
            {
                "item_type": "decision",
                "text": "We decided to use PostgreSQL",
                "confidence": 0.9,
                "char_offset": 10,
                "stakeholder": None,
            },
            {
                "item_type": "pain_point",
                "text": "Current system is too slow",
                "confidence": 0.8,
                "char_offset": 50,
                "stakeholder": "engineering team",
            },
            {
                "item_type": "feature_request",
                "text": "Need a dashboard for metrics",
                "confidence": 0.85,
                "char_offset": 100,
                "stakeholder": "product manager",
            },
        ])
        llm_tool = MockLLMTool(responses=[llm_response])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        doc = _make_doc(
            "meeting.txt",
            b"We discussed PostgreSQL and the team mentioned performance issues. PM requested a dashboard.",
            DocumentType.TRANSCRIPT_TXT,
        )
        result = await parser.parse_documents([doc])

        assert isinstance(result, ExtractionResult)
        assert len(result.items) == 3
        assert result.items[0].item_type == "decision"
        assert result.items[0].text == "We decided to use PostgreSQL"
        assert result.items[0].confidence == 0.9
        assert result.items[0].source_chunk_index == 0
        assert result.items[1].item_type == "pain_point"
        assert result.items[1].stakeholder == "engineering team"
        assert result.items[2].item_type == "feature_request"
        assert result.items[2].stakeholder == "product manager"

    @pytest.mark.asyncio
    async def test_architecture_extraction_returns_items(self) -> None:
        """Validates: Requirements 3.4"""
        llm_response = json.dumps([
            {
                "item_type": "constraint",
                "text": "System must handle 10k requests per second",
                "confidence": 0.95,
                "char_offset": 5,
                "section_heading": "Performance Requirements",
                "type_classification": "constraint",
            },
            {
                "item_type": "constraint",
                "text": "Use event-driven architecture for async processing",
                "confidence": 0.88,
                "char_offset": 80,
                "section_heading": "Architecture",
                "type_classification": "decision",
            },
        ])
        llm_tool = MockLLMTool(responses=[llm_response])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        html_content = b"<h1>Architecture</h1><p>System must handle 10k requests per second.</p><h2>Design</h2><p>Use event-driven architecture.</p>"
        doc = _make_doc("arch.html", html_content, DocumentType.ARCHITECTURE_HTML)
        result = await parser.parse_documents([doc])

        assert isinstance(result, ExtractionResult)
        assert len(result.items) == 2
        assert result.items[0].item_type == "constraint"
        assert result.items[0].section_heading == "Performance Requirements"
        assert result.items[0].type_classification == "constraint"
        assert result.items[1].type_classification == "decision"

    @pytest.mark.asyncio
    async def test_empty_extraction_returns_metadata_note(self) -> None:
        """Validates: Requirements 3.6"""
        llm_tool = MockLLMTool(responses=["[]"])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        doc = _make_doc(
            "empty_meeting.txt",
            b"Just some casual conversation with no actionable items.",
            DocumentType.TRANSCRIPT_TXT,
        )
        result = await parser.parse_documents([doc])

        assert isinstance(result, ExtractionResult)
        assert len(result.items) == 0
        assert "note" in result.metadata
        assert "no extractable items" in result.metadata["note"].lower() or "yielded no" in result.metadata["note"].lower()

    @pytest.mark.asyncio
    async def test_ingestion_failure_produces_error(self) -> None:
        """Test that ingestion failures are captured in errors list."""
        llm_tool = MockLLMTool(responses=["[]"])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        doc = _make_doc("bad.txt", b"\xff\xfe invalid utf8", DocumentType.TRANSCRIPT_TXT)
        result = await parser.parse_documents([doc])

        assert len(result.errors) == 1
        assert result.errors[0].filename == "bad.txt"
        assert result.errors[0].reason != ""

    @pytest.mark.asyncio
    async def test_multiple_documents_combined(self) -> None:
        """Test processing multiple documents in one call."""
        transcript_response = json.dumps([
            {"item_type": "decision", "text": "Use microservices", "confidence": 0.9, "char_offset": 0, "stakeholder": None}
        ])
        architecture_response = json.dumps([
            {"item_type": "constraint", "text": "Max 5 services", "confidence": 0.85, "char_offset": 0, "section_heading": "Constraints", "type_classification": "constraint"}
        ])
        llm_tool = MockLLMTool(responses=[transcript_response, architecture_response])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        docs = [
            _make_doc("meeting.txt", b"We decided on microservices.", DocumentType.TRANSCRIPT_TXT),
            _make_doc("arch.html", b"<h1>Constraints</h1><p>Max 5 services</p>", DocumentType.ARCHITECTURE_HTML),
        ]
        result = await parser.parse_documents(docs)

        assert len(result.items) == 2
        assert result.items[0].item_type == "decision"
        assert result.items[1].item_type == "constraint"
        assert len(result.errors) == 0
        # metadata note should NOT be set since items were found
        assert "note" not in result.metadata

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self) -> None:
        """Test that some docs can fail while others succeed."""
        llm_response = json.dumps([
            {"item_type": "decision", "text": "Use Redis for caching", "confidence": 0.92, "char_offset": 0, "stakeholder": None}
        ])
        llm_tool = MockLLMTool(responses=[llm_response])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        docs = [
            _make_doc("bad.txt", b"\xff\xfe broken", DocumentType.TRANSCRIPT_TXT),
            _make_doc("good.txt", b"We decided to use Redis for caching.", DocumentType.TRANSCRIPT_TXT),
        ]
        result = await parser.parse_documents(docs)

        assert len(result.items) == 1
        assert result.items[0].text == "Use Redis for caching"
        assert len(result.errors) == 1
        assert result.errors[0].filename == "bad.txt"


# ---------- _extract_from_transcript Tests ----------


class TestExtractFromTranscript:
    """Tests for the _extract_from_transcript method."""

    def test_extracts_decisions(self) -> None:
        """Validates: Requirements 3.1"""
        llm_response = json.dumps([
            {"item_type": "decision", "text": "Adopt Python 3.11", "confidence": 0.95, "char_offset": 0, "stakeholder": None}
        ])
        llm_tool = MockLLMTool(responses=[llm_response])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="We decided to adopt Python 3.11 for the project.", token_count=10)]
        items = parser._extract_from_transcript(chunks)

        assert len(items) == 1
        assert items[0].item_type == "decision"
        assert items[0].text == "Adopt Python 3.11"
        assert items[0].confidence == 0.95
        assert items[0].source_chunk_index == 0

    def test_extracts_pain_points_with_stakeholder(self) -> None:
        """Validates: Requirements 3.2"""
        llm_response = json.dumps([
            {"item_type": "pain_point", "text": "Deployment takes too long", "confidence": 0.8, "char_offset": 5, "stakeholder": "DevOps team"}
        ])
        llm_tool = MockLLMTool(responses=[llm_response])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=2, text="DevOps mentioned deployment takes too long.", token_count=7)]
        items = parser._extract_from_transcript(chunks)

        assert len(items) == 1
        assert items[0].item_type == "pain_point"
        assert items[0].stakeholder == "DevOps team"
        assert items[0].source_chunk_index == 2
        assert items[0].char_offset == 5

    def test_extracts_feature_requests(self) -> None:
        """Validates: Requirements 3.3"""
        llm_response = json.dumps([
            {"item_type": "feature_request", "text": "Add export to CSV", "confidence": 0.87, "char_offset": 20, "stakeholder": "sales team"}
        ])
        llm_tool = MockLLMTool(responses=[llm_response])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=1, text="Sales team asked for CSV export capability.", token_count=8)]
        items = parser._extract_from_transcript(chunks)

        assert len(items) == 1
        assert items[0].item_type == "feature_request"
        assert items[0].stakeholder == "sales team"

    def test_handles_invalid_json_gracefully(self) -> None:
        """LLM returning invalid JSON should not crash, just skip the chunk."""
        llm_tool = MockLLMTool(responses=["This is not JSON at all"])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="Some meeting text.", token_count=3)]
        items = parser._extract_from_transcript(chunks)

        assert items == []

    def test_handles_llm_tool_error_gracefully(self) -> None:
        """ToolError from LLM should not crash, just skip the chunk."""
        llm_tool = MockLLMTool(error=TransientToolError("API rate limit"))
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="Some meeting text.", token_count=3)]
        items = parser._extract_from_transcript(chunks)

        assert items == []

    def test_handles_non_array_json_response(self) -> None:
        """LLM returning a JSON object instead of array should be skipped."""
        llm_tool = MockLLMTool(responses=['{"item_type": "decision", "text": "something"}'])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="Some meeting text.", token_count=3)]
        items = parser._extract_from_transcript(chunks)

        assert items == []

    def test_skips_entries_with_empty_text(self) -> None:
        """Items with empty text should be filtered out."""
        llm_response = json.dumps([
            {"item_type": "decision", "text": "", "confidence": 0.9, "char_offset": 0, "stakeholder": None},
            {"item_type": "decision", "text": "Valid decision", "confidence": 0.8, "char_offset": 10, "stakeholder": None},
        ])
        llm_tool = MockLLMTool(responses=[llm_response])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="Some text.", token_count=2)]
        items = parser._extract_from_transcript(chunks)

        assert len(items) == 1
        assert items[0].text == "Valid decision"

    def test_multiple_chunks_each_processed(self) -> None:
        """Each chunk should be sent to the LLM independently."""
        response1 = json.dumps([
            {"item_type": "decision", "text": "Decision from chunk 0", "confidence": 0.9, "char_offset": 0, "stakeholder": None}
        ])
        response2 = json.dumps([
            {"item_type": "pain_point", "text": "Pain from chunk 1", "confidence": 0.7, "char_offset": 0, "stakeholder": "users"}
        ])
        llm_tool = MockLLMTool(responses=[response1, response2])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [
            TextChunk(index=0, text="First chunk of meeting notes.", token_count=5),
            TextChunk(index=1, text="Second chunk of meeting notes.", token_count=5),
        ]
        items = parser._extract_from_transcript(chunks)

        assert len(items) == 2
        assert items[0].source_chunk_index == 0
        assert items[1].source_chunk_index == 1
        assert len(llm_tool.calls) == 2

    def test_invalid_confidence_value_skipped(self) -> None:
        """Confidence outside [0.0, 1.0] should cause the entry to be skipped."""
        llm_response = json.dumps([
            {"item_type": "decision", "text": "Bad confidence", "confidence": 1.5, "char_offset": 0, "stakeholder": None},
            {"item_type": "decision", "text": "Good item", "confidence": 0.8, "char_offset": 10, "stakeholder": None},
        ])
        llm_tool = MockLLMTool(responses=[llm_response])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="Some text.", token_count=2)]
        items = parser._extract_from_transcript(chunks)

        # The item with confidence > 1.0 should be skipped due to Pydantic validation
        assert len(items) == 1
        assert items[0].text == "Good item"


# ---------- _extract_from_architecture Tests ----------


class TestExtractFromArchitecture:
    """Tests for the _extract_from_architecture method."""

    def test_extracts_constraints(self) -> None:
        """Validates: Requirements 3.4"""
        llm_response = json.dumps([
            {
                "item_type": "constraint",
                "text": "All services must use HTTPS",
                "confidence": 0.92,
                "char_offset": 0,
                "section_heading": "Security",
                "type_classification": "constraint",
            }
        ])
        llm_tool = MockLLMTool(responses=[llm_response])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="Security: All services must use HTTPS.", token_count=7)]
        items = parser._extract_from_architecture(chunks)

        assert len(items) == 1
        assert items[0].item_type == "constraint"
        assert items[0].text == "All services must use HTTPS"
        assert items[0].section_heading == "Security"
        assert items[0].type_classification == "constraint"
        assert items[0].confidence == 0.92

    def test_extracts_decisions(self) -> None:
        """Validates: Requirements 3.4"""
        llm_response = json.dumps([
            {
                "item_type": "constraint",
                "text": "Use PostgreSQL for primary data store",
                "confidence": 0.88,
                "char_offset": 15,
                "section_heading": "Data Layer",
                "type_classification": "decision",
            }
        ])
        llm_tool = MockLLMTool(responses=[llm_response])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="Data Layer: Use PostgreSQL for primary data store.", token_count=9)]
        items = parser._extract_from_architecture(chunks)

        assert len(items) == 1
        assert items[0].type_classification == "decision"
        assert items[0].section_heading == "Data Layer"

    def test_extracts_principles(self) -> None:
        """Validates: Requirements 3.4"""
        llm_response = json.dumps([
            {
                "item_type": "constraint",
                "text": "Prefer composition over inheritance",
                "confidence": 0.75,
                "char_offset": 0,
                "section_heading": "Design Principles",
                "type_classification": "principle",
            }
        ])
        llm_tool = MockLLMTool(responses=[llm_response])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="Design Principles: Prefer composition over inheritance.", token_count=6)]
        items = parser._extract_from_architecture(chunks)

        assert len(items) == 1
        assert items[0].type_classification == "principle"

    def test_handles_llm_error_gracefully(self) -> None:
        """ToolError should not crash architecture extraction."""
        llm_tool = MockLLMTool(error=ToolError("Service unavailable"))
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="Architecture doc content.", token_count=4)]
        items = parser._extract_from_architecture(chunks)

        assert items == []

    def test_handles_invalid_json_gracefully(self) -> None:
        """Invalid JSON from LLM should be skipped."""
        llm_tool = MockLLMTool(responses=["not valid json {{{"])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="Architecture doc content.", token_count=4)]
        items = parser._extract_from_architecture(chunks)

        assert items == []

    def test_skips_entries_with_empty_text(self) -> None:
        """Items with empty text should be filtered out."""
        llm_response = json.dumps([
            {"item_type": "constraint", "text": "", "confidence": 0.9, "char_offset": 0, "section_heading": "X", "type_classification": "constraint"},
            {"item_type": "constraint", "text": "Valid constraint", "confidence": 0.85, "char_offset": 10, "section_heading": "Y", "type_classification": "decision"},
        ])
        llm_tool = MockLLMTool(responses=[llm_response])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="Some arch text.", token_count=3)]
        items = parser._extract_from_architecture(chunks)

        assert len(items) == 1
        assert items[0].text == "Valid constraint"

    def test_multiple_chunks_processed_independently(self) -> None:
        """Each architecture chunk should be processed by the LLM separately."""
        resp1 = json.dumps([
            {"item_type": "constraint", "text": "Item from chunk 0", "confidence": 0.9, "char_offset": 0, "section_heading": "S1", "type_classification": "constraint"}
        ])
        resp2 = json.dumps([
            {"item_type": "constraint", "text": "Item from chunk 1", "confidence": 0.8, "char_offset": 0, "section_heading": "S2", "type_classification": "decision"}
        ])
        llm_tool = MockLLMTool(responses=[resp1, resp2])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [
            TextChunk(index=0, text="First architecture section.", token_count=4),
            TextChunk(index=1, text="Second architecture section.", token_count=4),
        ]
        items = parser._extract_from_architecture(chunks)

        assert len(items) == 2
        assert items[0].source_chunk_index == 0
        assert items[1].source_chunk_index == 1
        assert len(llm_tool.calls) == 2


# ---------- LLM Prompt Structure Tests ----------


class TestLLMPromptStructure:
    """Tests that verify the LLM is called with proper prompts."""

    def test_transcript_extraction_uses_system_prompt(self) -> None:
        """Verify the LLM is called with a system prompt for transcript extraction."""
        llm_tool = MockLLMTool(responses=["[]"])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="Meeting content here.", token_count=3)]
        parser._extract_from_transcript(chunks)

        assert len(llm_tool.calls) == 1
        assert llm_tool.calls[0]["system_prompt"] is not None
        assert "decision" in llm_tool.calls[0]["system_prompt"].lower()
        assert "pain_point" in llm_tool.calls[0]["system_prompt"].lower()
        assert "feature_request" in llm_tool.calls[0]["system_prompt"].lower()

    def test_architecture_extraction_uses_system_prompt(self) -> None:
        """Verify the LLM is called with a system prompt for architecture extraction."""
        llm_tool = MockLLMTool(responses=["[]"])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="Architecture content here.", token_count=3)]
        parser._extract_from_architecture(chunks)

        assert len(llm_tool.calls) == 1
        assert llm_tool.calls[0]["system_prompt"] is not None
        assert "constraint" in llm_tool.calls[0]["system_prompt"].lower()
        assert "decision" in llm_tool.calls[0]["system_prompt"].lower()
        assert "type_classification" in llm_tool.calls[0]["system_prompt"].lower()

    def test_transcript_prompt_includes_chunk_text(self) -> None:
        """Verify the user prompt includes the actual chunk text."""
        llm_tool = MockLLMTool(responses=["[]"])
        parser = ParserAgent(MockParsingTool(), llm_tool)

        from backlog_synthesizer.models.extraction import TextChunk
        chunks = [TextChunk(index=0, text="Specific meeting content about databases.", token_count=5)]
        parser._extract_from_transcript(chunks)

        assert "Specific meeting content about databases." in llm_tool.calls[0]["prompt"]
