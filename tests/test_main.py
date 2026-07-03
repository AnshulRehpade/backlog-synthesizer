"""Unit tests for the main entry point module.

Tests verify:
- Component wiring (PipelineConfig → tools → agents → orchestrator)
- File extension to DocumentType mapping
- Session inputs built correctly from file paths
- JSON backlog ticket parsing
- The `run` function returns serialized StoryOutput JSON
- Error handling for missing files and unsupported extensions
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backlog_synthesizer.main import (
    _build_session_inputs,
    _determine_document_type,
    _EXTENSION_MAP,
    create_pipeline,
    run,
    run_async,
)
from backlog_synthesizer.models.inputs import DocumentType


# --- Test: Extension to DocumentType Mapping ---


class TestDetermineDocumentType:
    """Tests for file extension to DocumentType resolution."""

    def test_txt_extension(self, tmp_path):
        """`.txt` maps to TRANSCRIPT_TXT."""
        f = tmp_path / "meeting.txt"
        assert _determine_document_type(f) == DocumentType.TRANSCRIPT_TXT

    def test_md_extension(self, tmp_path):
        """`.md` maps to TRANSCRIPT_MD."""
        f = tmp_path / "notes.md"
        assert _determine_document_type(f) == DocumentType.TRANSCRIPT_MD

    def test_pdf_extension(self, tmp_path):
        """`.pdf` maps to TRANSCRIPT_PDF."""
        f = tmp_path / "report.pdf"
        assert _determine_document_type(f) == DocumentType.TRANSCRIPT_PDF

    def test_html_extension(self, tmp_path):
        """`.html` maps to ARCHITECTURE_HTML."""
        f = tmp_path / "arch.html"
        assert _determine_document_type(f) == DocumentType.ARCHITECTURE_HTML

    def test_json_extension(self, tmp_path):
        """`.json` maps to BACKLOG_JSON."""
        f = tmp_path / "backlog.json"
        assert _determine_document_type(f) == DocumentType.BACKLOG_JSON

    def test_unsupported_extension_raises(self, tmp_path):
        """Unsupported extensions raise ValueError."""
        f = tmp_path / "data.csv"
        with pytest.raises(ValueError, match="Unsupported file extension"):
            _determine_document_type(f)

    def test_case_insensitive_extension(self, tmp_path):
        """Extension matching is case-insensitive."""
        f = tmp_path / "notes.TXT"
        assert _determine_document_type(f) == DocumentType.TRANSCRIPT_TXT

    def test_uppercase_md(self, tmp_path):
        """`.MD` maps to TRANSCRIPT_MD."""
        f = tmp_path / "notes.MD"
        assert _determine_document_type(f) == DocumentType.TRANSCRIPT_MD


# --- Test: Build Session Inputs ---


class TestBuildSessionInputs:
    """Tests for building SessionInputs from file paths."""

    def test_single_text_file(self, tmp_path):
        """Single .txt file produces one InputDocument."""
        f = tmp_path / "meeting.txt"
        f.write_text("Meeting content here")

        inputs = _build_session_inputs([str(f)])

        assert len(inputs.documents) == 1
        assert inputs.documents[0].filename == "meeting.txt"
        assert inputs.documents[0].document_type == DocumentType.TRANSCRIPT_TXT
        assert inputs.documents[0].content == b"Meeting content here"
        assert inputs.documents[0].size_bytes == 20

    def test_multiple_files(self, tmp_path):
        """Multiple files produce multiple InputDocuments."""
        f1 = tmp_path / "meeting.txt"
        f1.write_text("Content 1")
        f2 = tmp_path / "arch.html"
        f2.write_text("<h1>Architecture</h1>")

        inputs = _build_session_inputs([str(f1), str(f2)])

        assert len(inputs.documents) == 2
        types = {d.document_type for d in inputs.documents}
        assert DocumentType.TRANSCRIPT_TXT in types
        assert DocumentType.ARCHITECTURE_HTML in types

    def test_json_file_parsed_as_tickets(self, tmp_path):
        """JSON files are parsed as backlog tickets, not added as documents."""
        tickets = [
            {
                "id": "TICKET-1",
                "title": "Feature A",
                "description": "Description A",
                "status": "open",
                "tags": ["feature"],
            },
            {
                "id": "TICKET-2",
                "title": "Bug B",
                "description": "Description B",
                "status": "closed",
            },
        ]
        f = tmp_path / "backlog.json"
        f.write_text(json.dumps(tickets))

        inputs = _build_session_inputs([str(f)])

        # JSON files should NOT appear in documents
        assert len(inputs.documents) == 0
        # Should appear as backlog tickets
        assert len(inputs.backlog_tickets) == 2
        assert inputs.backlog_tickets[0].id == "TICKET-1"
        assert inputs.backlog_tickets[1].id == "TICKET-2"

    def test_mixed_files(self, tmp_path):
        """Mix of text files and JSON produces documents + tickets."""
        txt = tmp_path / "meeting.txt"
        txt.write_text("Meeting notes")
        tickets = [
            {
                "id": "T-1",
                "title": "Existing",
                "description": "Existing feature",
                "status": "open",
            }
        ]
        json_file = tmp_path / "backlog.json"
        json_file.write_text(json.dumps(tickets))

        inputs = _build_session_inputs([str(txt), str(json_file)])

        assert len(inputs.documents) == 1
        assert len(inputs.backlog_tickets) == 1

    def test_invalid_ticket_skipped(self, tmp_path):
        """Invalid tickets in JSON are skipped, valid ones are kept."""
        tickets = [
            {
                "id": "TICKET-1",
                "title": "Valid",
                "description": "Valid description",
                "status": "open",
            },
            {"invalid": "no required fields"},
        ]
        f = tmp_path / "backlog.json"
        f.write_text(json.dumps(tickets))

        inputs = _build_session_inputs([str(f)])

        # Only valid ticket should be present
        assert len(inputs.backlog_tickets) == 1
        assert inputs.backlog_tickets[0].id == "TICKET-1"

    def test_session_id_generated(self, tmp_path):
        """Session ID is auto-generated as UUID."""
        f = tmp_path / "meeting.txt"
        f.write_text("content")

        inputs = _build_session_inputs([str(f)])

        assert inputs.session_id is not None
        assert len(inputs.session_id) > 0
        # UUID format check (has hyphens, correct length)
        assert len(inputs.session_id) == 36

    def test_file_not_found_raises(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _build_session_inputs(["/nonexistent/path/file.txt"])

    def test_unsupported_extension_raises(self, tmp_path):
        """Unsupported file extension raises ValueError."""
        f = tmp_path / "data.csv"
        f.write_text("a,b,c")

        with pytest.raises(ValueError, match="Unsupported file extension"):
            _build_session_inputs([str(f)])

    def test_path_objects_accepted(self, tmp_path):
        """Path objects are accepted in addition to strings."""
        f = tmp_path / "meeting.txt"
        f.write_text("content")

        inputs = _build_session_inputs([f])

        assert len(inputs.documents) == 1

    def test_json_non_array_skipped(self, tmp_path):
        """Non-array JSON content is skipped with a warning."""
        f = tmp_path / "backlog.json"
        f.write_text('{"not": "an array"}')

        inputs = _build_session_inputs([str(f)])

        assert len(inputs.backlog_tickets) == 0
        assert len(inputs.documents) == 0


# --- Test: Create Pipeline ---


class TestCreatePipeline:
    """Tests for the create_pipeline function."""

    @patch("backlog_synthesizer.main.PipelineConfig.from_env")
    def test_create_pipeline_returns_orchestrator(self, mock_from_env):
        """create_pipeline returns an OrchestratorAgent instance."""
        mock_config = MagicMock()
        mock_config.create_document_parser.return_value = MagicMock()
        mock_config.create_embedding_tool.return_value = MagicMock()
        mock_config.create_vector_search_tool.return_value = MagicMock()
        mock_config.create_llm_tool.return_value = MagicMock()
        mock_config.tokenizer_model = "cl100k_base"
        mock_config.react_reasoning_enabled = True
        mock_from_env.return_value = mock_config

        orchestrator = create_pipeline()

        assert orchestrator is not None
        assert hasattr(orchestrator, "run_session")

    def test_create_pipeline_with_explicit_config(self):
        """create_pipeline works with an explicit PipelineConfig."""
        mock_config = MagicMock()
        mock_config.create_document_parser.return_value = MagicMock()
        mock_config.create_embedding_tool.return_value = MagicMock()
        mock_config.create_vector_search_tool.return_value = MagicMock()
        mock_config.create_llm_tool.return_value = MagicMock()
        mock_config.tokenizer_model = "cl100k_base"
        mock_config.react_reasoning_enabled = True

        orchestrator = create_pipeline(config=mock_config)

        assert orchestrator is not None
        # Verify tools were created from config
        mock_config.create_document_parser.assert_called_once()
        mock_config.create_embedding_tool.assert_called_once()
        mock_config.create_vector_search_tool.assert_called_once()
        mock_config.create_llm_tool.assert_called_once()


# --- Test: Run Function ---


class TestRunFunction:
    """Tests for the run and run_async functions."""

    @pytest.mark.asyncio
    async def test_run_async_returns_json(self, tmp_path):
        """run_async returns valid JSON string on success."""
        # Create a test file
        f = tmp_path / "meeting.txt"
        f.write_text("We decided to use Python for the project")

        # Mock the pipeline to avoid real LLM/embedding calls
        with patch("backlog_synthesizer.main.create_pipeline") as mock_create:
            mock_orchestrator = MagicMock()
            mock_result = MagicMock()
            mock_result.output = MagicMock()
            mock_result.output.model_dump_json.return_value = '{"epics": [], "index": [], "metadata": {"session_id": "test", "timestamp": "2024-01-01T00:00:00Z"}}'
            mock_orchestrator.run_session = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_orchestrator

            result = await run_async([str(f)])

            assert isinstance(result, str)
            parsed = json.loads(result)
            assert "epics" in parsed

    @pytest.mark.asyncio
    async def test_run_async_returns_error_json_on_failure(self, tmp_path):
        """run_async returns error JSON when pipeline produces no output."""
        f = tmp_path / "meeting.txt"
        f.write_text("content")

        with patch("backlog_synthesizer.main.create_pipeline") as mock_create:
            mock_orchestrator = MagicMock()
            mock_result = MagicMock()
            mock_result.output = None
            mock_result.session_id = "test-session"
            mock_result.status = "partial_failure"
            mock_result.errors = [{"step": "parser", "error": "some error"}]
            mock_orchestrator.run_session = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_orchestrator

            result = await run_async([str(f)])

            parsed = json.loads(result)
            assert parsed["status"] == "partial_failure"
            assert parsed["session_id"] == "test-session"
            assert len(parsed["errors"]) == 1

    def test_run_sync_wrapper(self, tmp_path):
        """run() synchronous wrapper works correctly."""
        f = tmp_path / "meeting.txt"
        f.write_text("content")

        with patch("backlog_synthesizer.main.create_pipeline") as mock_create:
            mock_orchestrator = MagicMock()
            mock_result = MagicMock()
            mock_result.output = MagicMock()
            mock_result.output.model_dump_json.return_value = '{"test": true}'
            mock_orchestrator.run_session = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_orchestrator

            result = run([str(f)])

            assert isinstance(result, str)
            assert json.loads(result) == {"test": True}


# --- Test: Extension Map Completeness ---


class TestExtensionMap:
    """Tests verifying the extension map covers all expected types."""

    def test_all_document_types_covered(self):
        """Every DocumentType has at least one extension mapping to it."""
        mapped_types = set(_EXTENSION_MAP.values())
        for doc_type in DocumentType:
            assert doc_type in mapped_types, (
                f"DocumentType.{doc_type.name} has no file extension mapping"
            )

    def test_extension_map_has_expected_entries(self):
        """The extension map has exactly the expected number of entries."""
        assert len(_EXTENSION_MAP) == 5
