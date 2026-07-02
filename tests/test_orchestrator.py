"""Unit tests for the Orchestrator Agent pipeline sequencing and session management.

Tests verify:
- Pipeline invocation ordering (Parser → Gap Detection → Story Writer)
- Session state creation and intermediate result storage
- Backlog ticket validation (valid accepted, invalid rejected)
- Error handling when sub-agents fail
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from backlog_synthesizer.agents.orchestrator import OrchestratorAgent, SessionResult
from backlog_synthesizer.models.extraction import ExtractedItem, ExtractionResult
from backlog_synthesizer.models.gap_detection import GapReport, GapReportEntry
from backlog_synthesizer.models.inputs import (
    BacklogTicket,
    DocumentType,
    InputDocument,
    SessionInputs,
)
from backlog_synthesizer.models.output import (
    AcceptanceCriterion,
    Epic,
    OutputMetadata,
    StoryOutput,
    UserStory,
)
from backlog_synthesizer.agents.story_writer import SerializationResult


# --- Fixtures ---


@pytest.fixture
def mock_memory():
    """Create a mock MemoryEngine."""
    memory = MagicMock()
    memory.store_intermediate = MagicMock()
    memory.store_for_search = MagicMock()
    memory.log_action = MagicMock()
    return memory


@pytest.fixture
def sample_extracted_items():
    """Create sample extracted items for testing."""
    return [
        ExtractedItem(
            item_type="feature_request",
            text="Add dark mode support for the dashboard",
            source_chunk_index=0,
            confidence=0.9,
            stakeholder="user",
        ),
        ExtractedItem(
            item_type="decision",
            text="Use PostgreSQL for the main database",
            source_chunk_index=1,
            confidence=0.85,
        ),
    ]


@pytest.fixture
def sample_extraction_result(sample_extracted_items):
    """Create a sample ExtractionResult."""
    return ExtractionResult(items=sample_extracted_items, errors=[])


@pytest.fixture
def sample_gap_report(sample_extracted_items):
    """Create a sample GapReport."""
    entries = [
        GapReportEntry(item=item, classification="new", confidence=1.0)
        for item in sample_extracted_items
    ]
    return GapReport(
        entries=entries,
        total_new=2,
        total_duplicates=0,
        total_conflicts=0,
        total_unprocessed=0,
    )


@pytest.fixture
def sample_stories():
    """Create sample UserStory objects."""
    return [
        UserStory(
            title="Dark Mode Support",
            user_story="As a user, I want dark mode, so that I can reduce eye strain",
            acceptance_criteria=[
                AcceptanceCriterion(description="Toggle switch in settings"),
                AcceptanceCriterion(description="Colors conform to WCAG standards"),
            ],
            tags=["ui", "accessibility"],
        ),
        UserStory(
            title="Database Migration",
            user_story="As a developer, I want PostgreSQL, so that I have reliable storage",
            acceptance_criteria=[
                AcceptanceCriterion(description="Migration script runs without errors"),
                AcceptanceCriterion(description="All existing data preserved"),
            ],
            tags=["backend", "database"],
        ),
    ]


@pytest.fixture
def sample_epics(sample_stories):
    """Create sample Epic objects."""
    return [
        Epic(epic_title="UI Improvements", stories=[sample_stories[0]]),
        Epic(epic_title="Backend Infrastructure", stories=[sample_stories[1]]),
    ]


@pytest.fixture
def sample_story_output(sample_epics):
    """Create a sample StoryOutput."""
    return StoryOutput(
        index=[
            {"epic_title": "UI Improvements", "story_count": 1},
            {"epic_title": "Backend Infrastructure", "story_count": 1},
        ],
        epics=sample_epics,
        metadata=OutputMetadata(
            session_id="test-session-1",
            timestamp=datetime.now(timezone.utc),
        ),
    )


@pytest.fixture
def sample_session_inputs():
    """Create sample SessionInputs."""
    return SessionInputs(
        session_id="test-session-1",
        documents=[
            InputDocument(
                filename="meeting.txt",
                document_type=DocumentType.TRANSCRIPT_TXT,
                content=b"Meeting notes content here",
                size_bytes=26,
            ),
        ],
        backlog_tickets=[
            BacklogTicket(
                id="TICKET-1",
                title="Existing feature",
                description="An existing backlog item",
                status="open",
                tags=["feature"],
            ),
        ],
    )


@pytest.fixture
def mock_parser(sample_extraction_result):
    """Create a mock ParserAgent."""
    parser = MagicMock()
    parser.parse_documents = AsyncMock(return_value=sample_extraction_result)
    return parser


@pytest.fixture
def mock_gap_detector(sample_gap_report):
    """Create a mock GapDetectionAgent."""
    detector = MagicMock()
    detector.analyze_gaps = AsyncMock(return_value=sample_gap_report)
    return detector


@pytest.fixture
def mock_story_writer(sample_stories, sample_epics, sample_story_output):
    """Create a mock StoryWriterAgent."""
    writer = MagicMock()
    writer.generate_stories = AsyncMock(return_value=sample_stories)
    writer.group_into_epics = MagicMock(return_value=sample_epics)
    writer.serialize_output = MagicMock(
        return_value=SerializationResult(output=sample_story_output, errors=[])
    )
    return writer


@pytest.fixture
def orchestrator(mock_parser, mock_gap_detector, mock_story_writer, mock_memory):
    """Create an OrchestratorAgent with mocked dependencies."""
    return OrchestratorAgent(
        parser=mock_parser,
        gap_detector=mock_gap_detector,
        story_writer=mock_story_writer,
        memory=mock_memory,
    )


# --- Test: Pipeline Ordering ---


class TestPipelineOrdering:
    """Tests verifying the pipeline invocation order: Parser → Gap Detection → Story Writer."""

    async def test_parser_invoked_first(
        self, orchestrator, mock_parser, sample_session_inputs
    ):
        """Parser Agent is invoked before Gap Detection and Story Writer."""
        call_order = []

        async def track_parser(*args, **kwargs):
            call_order.append("parser")
            return orchestrator._parser.parse_documents.return_value

        async def track_gap(*args, **kwargs):
            call_order.append("gap_detection")
            return orchestrator._gap_detector.analyze_gaps.return_value

        async def track_story(*args, **kwargs):
            call_order.append("story_writer")
            return orchestrator._story_writer.generate_stories.return_value

        mock_parser.parse_documents = AsyncMock(side_effect=track_parser)
        orchestrator._parser = mock_parser
        orchestrator._gap_detector.analyze_gaps = AsyncMock(side_effect=track_gap)
        orchestrator._story_writer.generate_stories = AsyncMock(side_effect=track_story)

        await orchestrator.run_session(sample_session_inputs)

        assert call_order == ["parser", "gap_detection", "story_writer"]

    async def test_gap_detection_receives_parser_output(
        self, orchestrator, mock_parser, mock_gap_detector, sample_session_inputs,
        sample_extracted_items,
    ):
        """Gap Detection Agent receives the extracted items from the Parser."""
        await orchestrator.run_session(sample_session_inputs)

        mock_gap_detector.analyze_gaps.assert_called_once_with(sample_extracted_items)

    async def test_story_writer_receives_gap_report_entries(
        self, orchestrator, mock_story_writer, sample_session_inputs, sample_gap_report
    ):
        """Story Writer Agent receives entries from the Gap Report."""
        await orchestrator.run_session(sample_session_inputs)

        mock_story_writer.generate_stories.assert_called_once_with(
            sample_gap_report.entries
        )

    async def test_full_pipeline_completes_successfully(
        self, orchestrator, sample_session_inputs
    ):
        """Full pipeline returns 'completed' status when all agents succeed."""
        result = await orchestrator.run_session(sample_session_inputs)

        assert result.status == "completed"
        assert result.output is not None
        assert result.session_id == "test-session-1"


# --- Test: Session Management ---


class TestSessionManagement:
    """Tests verifying session creation and intermediate result storage."""

    async def test_session_state_created_at_start(
        self, orchestrator, mock_memory, sample_session_inputs
    ):
        """Session state is stored at the beginning of the pipeline."""
        await orchestrator.run_session(sample_session_inputs)

        # First store_intermediate call should be the initial session state
        first_call = mock_memory.store_intermediate.call_args_list[0]
        assert first_call[0][0] == "test-session-1"
        assert first_call[0][1] == "session_state"
        stored_state = first_call[0][2]
        assert stored_state["session_id"] == "test-session-1"
        assert stored_state["status"] == "in_progress"

    async def test_extraction_result_stored_after_parser(
        self, orchestrator, mock_memory, sample_session_inputs
    ):
        """Extraction result is stored in memory after Parser completes."""
        await orchestrator.run_session(sample_session_inputs)

        # Find the store_intermediate call for extraction_result
        stored_keys = [
            call[0][1] for call in mock_memory.store_intermediate.call_args_list
        ]
        assert "extraction_result" in stored_keys

    async def test_gap_report_stored_after_gap_detection(
        self, orchestrator, mock_memory, sample_session_inputs
    ):
        """Gap report is stored in memory after Gap Detection completes."""
        await orchestrator.run_session(sample_session_inputs)

        stored_keys = [
            call[0][1] for call in mock_memory.store_intermediate.call_args_list
        ]
        assert "gap_report" in stored_keys

    async def test_story_output_stored_after_story_writer(
        self, orchestrator, mock_memory, sample_session_inputs
    ):
        """Story output is stored in memory after Story Writer completes."""
        await orchestrator.run_session(sample_session_inputs)

        stored_keys = [
            call[0][1] for call in mock_memory.store_intermediate.call_args_list
        ]
        assert "story_output" in stored_keys

    async def test_intermediate_storage_order(
        self, orchestrator, mock_memory, sample_session_inputs
    ):
        """Intermediate results are stored in the correct order."""
        await orchestrator.run_session(sample_session_inputs)

        stored_keys = [
            call[0][1] for call in mock_memory.store_intermediate.call_args_list
        ]
        # session_state first, then extraction_result, gap_report, story_output, final session_state
        assert stored_keys[0] == "session_state"
        er_idx = stored_keys.index("extraction_result")
        gr_idx = stored_keys.index("gap_report")
        so_idx = stored_keys.index("story_output")
        assert er_idx < gr_idx < so_idx

    async def test_audit_log_entries_created_for_each_agent(
        self, orchestrator, mock_memory, sample_session_inputs
    ):
        """Audit log entries are created for each sub-agent invocation."""
        await orchestrator.run_session(sample_session_inputs)

        # log_action should be called for Parser, Gap Detection, Story Writer
        assert mock_memory.log_action.call_count >= 3
        agent_names = [
            call[0][1].agent_name
            for call in mock_memory.log_action.call_args_list
        ]
        assert "ParserAgent" in agent_names
        assert "GapDetectionAgent" in agent_names
        assert "StoryWriterAgent" in agent_names

    async def test_extracted_items_stored_for_search(
        self, orchestrator, mock_memory, sample_session_inputs
    ):
        """Extracted items are stored in Long-Term Memory for future search."""
        await orchestrator.run_session(sample_session_inputs)

        # store_for_search should be called (for backlog tickets + extracted items)
        assert mock_memory.store_for_search.call_count >= 1


# --- Test: Backlog Ticket Validation ---


class TestBacklogTicketValidation:
    """Tests verifying backlog ticket JSON schema validation."""

    async def test_valid_tickets_loaded_into_memory(
        self, orchestrator, mock_memory, sample_session_inputs
    ):
        """Valid backlog tickets are loaded into Long-Term Memory."""
        await orchestrator.run_session(sample_session_inputs)

        # store_for_search should be called with ticket data
        store_calls = mock_memory.store_for_search.call_args_list
        # First call should be for backlog tickets
        first_call = store_calls[0]
        items = first_call[0][1]
        assert any("TICKET-1" in item["item_id"] for item in items)

    async def test_session_with_no_tickets_continues(
        self, orchestrator, mock_memory
    ):
        """Pipeline continues normally when no backlog tickets are provided."""
        inputs = SessionInputs(
            session_id="test-session-2",
            documents=[
                InputDocument(
                    filename="notes.txt",
                    document_type=DocumentType.TRANSCRIPT_TXT,
                    content=b"Some content",
                    size_bytes=12,
                ),
            ],
            backlog_tickets=[],
        )

        result = await orchestrator.run_session(inputs)

        assert result.status == "completed"

    async def test_backlog_documents_filtered_from_parser(
        self, orchestrator, mock_parser
    ):
        """Backlog JSON documents are not passed to the Parser Agent."""
        inputs = SessionInputs(
            session_id="test-session-3",
            documents=[
                InputDocument(
                    filename="meeting.txt",
                    document_type=DocumentType.TRANSCRIPT_TXT,
                    content=b"Meeting notes",
                    size_bytes=13,
                ),
                InputDocument(
                    filename="backlog.json",
                    document_type=DocumentType.BACKLOG_JSON,
                    content=b'[{"id": "1"}]',
                    size_bytes=14,
                ),
            ],
            backlog_tickets=[],
        )

        await orchestrator.run_session(inputs)

        # Parser should only receive the non-backlog document
        call_args = mock_parser.parse_documents.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].document_type == DocumentType.TRANSCRIPT_TXT


# --- Test: Error Handling ---


class TestErrorHandling:
    """Tests verifying error handling when sub-agents fail."""

    async def test_parser_failure_returns_partial_failure(
        self, mock_gap_detector, mock_story_writer, mock_memory, sample_session_inputs
    ):
        """Pipeline returns partial_failure when Parser Agent fails."""
        parser = MagicMock()
        parser.parse_documents = AsyncMock(side_effect=RuntimeError("Parse error"))

        orchestrator = OrchestratorAgent(
            parser=parser,
            gap_detector=mock_gap_detector,
            story_writer=mock_story_writer,
            memory=mock_memory,
        )

        result = await orchestrator.run_session(sample_session_inputs)

        assert result.status == "partial_failure"
        assert any("parser" in e.get("step", "") for e in result.errors)

    async def test_gap_detection_failure_returns_partial_failure(
        self, mock_parser, mock_story_writer, mock_memory, sample_session_inputs
    ):
        """Pipeline returns partial_failure when Gap Detection Agent fails."""
        gap_detector = MagicMock()
        gap_detector.analyze_gaps = AsyncMock(
            side_effect=RuntimeError("Gap detection error")
        )

        orchestrator = OrchestratorAgent(
            parser=mock_parser,
            gap_detector=gap_detector,
            story_writer=mock_story_writer,
            memory=mock_memory,
        )

        result = await orchestrator.run_session(sample_session_inputs)

        assert result.status == "partial_failure"
        assert any("gap_detection" in e.get("step", "") for e in result.errors)

    async def test_story_writer_failure_returns_partial_failure(
        self, mock_parser, mock_gap_detector, mock_memory, sample_session_inputs
    ):
        """Pipeline returns partial_failure when Story Writer Agent fails."""
        story_writer = MagicMock()
        story_writer.generate_stories = AsyncMock(
            side_effect=RuntimeError("Story writer error")
        )

        orchestrator = OrchestratorAgent(
            parser=mock_parser,
            gap_detector=mock_gap_detector,
            story_writer=story_writer,
            memory=mock_memory,
        )

        result = await orchestrator.run_session(sample_session_inputs)

        assert result.status == "partial_failure"
        assert any("story_writer" in e.get("step", "") for e in result.errors)

    async def test_parser_failure_stops_pipeline(
        self, mock_gap_detector, mock_story_writer, mock_memory, sample_session_inputs
    ):
        """When Parser fails, subsequent agents are not invoked."""
        parser = MagicMock()
        parser.parse_documents = AsyncMock(side_effect=RuntimeError("Parse error"))

        orchestrator = OrchestratorAgent(
            parser=parser,
            gap_detector=mock_gap_detector,
            story_writer=mock_story_writer,
            memory=mock_memory,
        )

        await orchestrator.run_session(sample_session_inputs)

        mock_gap_detector.analyze_gaps.assert_not_called()
        mock_story_writer.generate_stories.assert_not_called()

    async def test_gap_detection_failure_stops_story_writer(
        self, mock_parser, mock_story_writer, mock_memory, sample_session_inputs
    ):
        """When Gap Detection fails, Story Writer is not invoked."""
        gap_detector = MagicMock()
        gap_detector.analyze_gaps = AsyncMock(
            side_effect=RuntimeError("Gap detection error")
        )

        orchestrator = OrchestratorAgent(
            parser=mock_parser,
            gap_detector=gap_detector,
            story_writer=mock_story_writer,
            memory=mock_memory,
        )

        await orchestrator.run_session(sample_session_inputs)

        mock_story_writer.generate_stories.assert_not_called()

    async def test_session_state_updated_on_failure(
        self, mock_gap_detector, mock_story_writer, mock_memory, sample_session_inputs
    ):
        """Session state is updated with error status when pipeline fails."""
        parser = MagicMock()
        parser.parse_documents = AsyncMock(side_effect=RuntimeError("Parse error"))

        orchestrator = OrchestratorAgent(
            parser=parser,
            gap_detector=mock_gap_detector,
            story_writer=mock_story_writer,
            memory=mock_memory,
        )

        await orchestrator.run_session(sample_session_inputs)

        # Last store_intermediate call should update session_state
        last_session_call = None
        for call in mock_memory.store_intermediate.call_args_list:
            if call[0][1] == "session_state":
                last_session_call = call
        assert last_session_call is not None
        stored = last_session_call[0][2]
        assert stored["status"] == "partial_failure"
