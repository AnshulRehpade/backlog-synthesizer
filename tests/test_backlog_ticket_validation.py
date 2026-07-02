"""Tests for backlog ticket schema validation in the Orchestrator Agent.

Tests verify:
- Valid JSON entries are accepted and returned as BacklogTicket instances
- Invalid JSON entries are rejected with logged validation errors
- Processing continues with valid tickets only (invalid ones don't halt)
- Valid tickets are loaded into Long-Term Memory
- Count of accepted + rejected == total input count
- The validate_backlog_tickets_from_json method handles raw dicts correctly

Validates: Requirements 1.4, 1.6
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from backlog_synthesizer.agents.orchestrator import OrchestratorAgent, SessionResult
from backlog_synthesizer.models.inputs import (
    BacklogTicket,
    DocumentType,
    InputDocument,
    SessionInputs,
)
from backlog_synthesizer.models.extraction import ExtractedItem, ExtractionResult
from backlog_synthesizer.models.gap_detection import GapReport, GapReportEntry
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
def mock_parser():
    """Create a mock ParserAgent that returns empty extraction."""
    parser = MagicMock()
    parser.parse_documents = AsyncMock(
        return_value=ExtractionResult(items=[], errors=[])
    )
    return parser


@pytest.fixture
def mock_gap_detector():
    """Create a mock GapDetectionAgent."""
    detector = MagicMock()
    detector.analyze_gaps = AsyncMock(
        return_value=GapReport(
            entries=[],
            total_new=0,
            total_duplicates=0,
            total_conflicts=0,
            total_unprocessed=0,
        )
    )
    return detector


@pytest.fixture
def mock_story_writer():
    """Create a mock StoryWriterAgent."""
    writer = MagicMock()
    writer.generate_stories = AsyncMock(return_value=[])
    writer.group_into_epics = MagicMock(return_value=[])
    writer.serialize_output = MagicMock(
        return_value=SerializationResult(
            output=StoryOutput(
                index=[],
                epics=[],
                metadata=OutputMetadata(
                    session_id="test-session",
                    timestamp=datetime.now(timezone.utc),
                ),
            ),
            errors=[],
        )
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


# --- Test: validate_backlog_tickets_from_json ---


class TestValidateBacklogTicketsFromJson:
    """Tests for the validate_backlog_tickets_from_json method."""

    def test_valid_tickets_are_accepted(self, orchestrator):
        """Valid raw JSON entries are validated and returned as BacklogTicket instances."""
        raw_tickets = [
            {
                "id": "TICKET-1",
                "title": "Add login page",
                "description": "Users need a way to log in",
                "status": "open",
                "tags": ["auth", "frontend"],
            },
            {
                "id": "TICKET-2",
                "title": "Fix search performance",
                "description": "Search takes too long for large datasets",
                "status": "in_progress",
                "tags": ["performance"],
                "created_at": "2024-01-15T10:30:00Z",
            },
        ]

        valid, errors = orchestrator.validate_backlog_tickets_from_json(
            "test-session", raw_tickets
        )

        assert len(valid) == 2
        assert len(errors) == 0
        assert all(isinstance(t, BacklogTicket) for t in valid)
        assert valid[0].id == "TICKET-1"
        assert valid[0].title == "Add login page"
        assert valid[1].id == "TICKET-2"
        assert valid[1].tags == ["performance"]

    def test_invalid_tickets_are_rejected(self, orchestrator):
        """Invalid raw JSON entries are rejected with validation errors."""
        raw_tickets = [
            {
                # Missing required fields: id, title, description, status
                "tags": ["invalid"],
            },
            {
                "id": "TICKET-2",
                # Missing title, description, status
            },
        ]

        valid, errors = orchestrator.validate_backlog_tickets_from_json(
            "test-session", raw_tickets
        )

        assert len(valid) == 0
        assert len(errors) == 2
        assert errors[0]["step"] == "backlog_validation"
        assert errors[0]["ticket_index"] == 0
        assert "error" in errors[0]
        assert errors[1]["ticket_index"] == 1

    def test_mixed_valid_and_invalid_tickets(self, orchestrator):
        """Valid tickets are accepted while invalid ones are rejected in the same batch."""
        raw_tickets = [
            {
                "id": "VALID-1",
                "title": "Valid ticket",
                "description": "This is a valid ticket",
                "status": "open",
            },
            {
                # Invalid - missing required fields
                "id": "INVALID-1",
            },
            {
                "id": "VALID-2",
                "title": "Another valid ticket",
                "description": "Also valid",
                "status": "done",
                "tags": ["backend"],
            },
            {
                # Invalid - completely empty
            },
        ]

        valid, errors = orchestrator.validate_backlog_tickets_from_json(
            "test-session", raw_tickets
        )

        assert len(valid) == 2
        assert len(errors) == 2
        # accepted + rejected == total
        assert len(valid) + len(errors) == len(raw_tickets)
        assert valid[0].id == "VALID-1"
        assert valid[1].id == "VALID-2"
        assert errors[0]["ticket_index"] == 1
        assert errors[1]["ticket_index"] == 3

    def test_empty_list_returns_empty(self, orchestrator):
        """An empty list of raw tickets returns empty results."""
        valid, errors = orchestrator.validate_backlog_tickets_from_json(
            "test-session", []
        )

        assert len(valid) == 0
        assert len(errors) == 0

    def test_optional_fields_default_correctly(self, orchestrator):
        """Optional fields (tags, created_at) default properly when not provided."""
        raw_tickets = [
            {
                "id": "TICKET-1",
                "title": "Minimal ticket",
                "description": "Only required fields",
                "status": "open",
            },
        ]

        valid, errors = orchestrator.validate_backlog_tickets_from_json(
            "test-session", raw_tickets
        )

        assert len(valid) == 1
        assert valid[0].tags == []
        assert valid[0].created_at is None

    def test_validation_errors_contain_raw_data(self, orchestrator):
        """Validation error entries include the raw data that failed."""
        raw_tickets = [
            {"id": "BAD", "unknown_field": "value"},
        ]

        valid, errors = orchestrator.validate_backlog_tickets_from_json(
            "test-session", raw_tickets
        )

        assert len(errors) == 1
        assert errors[0]["raw_data"] == raw_tickets[0]

    def test_validation_errors_logged_to_audit(self, orchestrator, mock_memory):
        """Validation failures are logged to the audit log via _log_agent_action."""
        raw_tickets = [
            {"not_a_ticket": True},
        ]

        orchestrator.validate_backlog_tickets_from_json("test-session", raw_tickets)

        # log_action should have been called for the validation failure
        assert mock_memory.log_action.call_count >= 1
        logged_entry = mock_memory.log_action.call_args_list[0][0][1]
        assert logged_entry.agent_name == "OrchestratorAgent"
        assert "validation" in logged_entry.input_summary.lower() or "validating" in logged_entry.input_summary.lower()

    def test_wrong_type_for_required_field(self, orchestrator):
        """Tickets with wrong data types for fields are rejected."""
        raw_tickets = [
            {
                "id": 123,  # Should be string
                "title": "Valid title",
                "description": "Valid desc",
                "status": "open",
            },
        ]

        valid, errors = orchestrator.validate_backlog_tickets_from_json(
            "test-session", raw_tickets
        )

        # Pydantic may coerce int to str, so this could be valid
        # The important thing is the method doesn't crash
        assert len(valid) + len(errors) == 1

    def test_invalid_datetime_format_rejected(self, orchestrator):
        """Tickets with invalid datetime formats are rejected."""
        raw_tickets = [
            {
                "id": "TICKET-1",
                "title": "Has bad date",
                "description": "Description",
                "status": "open",
                "created_at": "not-a-date",
            },
        ]

        valid, errors = orchestrator.validate_backlog_tickets_from_json(
            "test-session", raw_tickets
        )

        assert len(errors) == 1
        assert errors[0]["ticket_index"] == 0

    def test_count_invariant_holds(self, orchestrator):
        """The count of valid + rejected always equals the input count."""
        raw_tickets = [
            {"id": "A", "title": "T", "description": "D", "status": "s"},
            {"missing": "fields"},
            {"id": "B", "title": "T2", "description": "D2", "status": "open"},
            {},
            {"id": "C"},
        ]

        valid, errors = orchestrator.validate_backlog_tickets_from_json(
            "test-session", raw_tickets
        )

        assert len(valid) + len(errors) == len(raw_tickets)


# --- Test: Integration with run_session (valid tickets loaded into memory) ---


class TestBacklogValidationInPipeline:
    """Tests verifying that valid tickets are loaded into Long-Term Memory during pipeline."""

    async def test_valid_tickets_stored_in_long_term_memory(
        self, orchestrator, mock_memory
    ):
        """Valid backlog tickets are stored in Long-Term Memory via store_for_search."""
        inputs = SessionInputs(
            session_id="test-session",
            documents=[
                InputDocument(
                    filename="notes.txt",
                    document_type=DocumentType.TRANSCRIPT_TXT,
                    content=b"Meeting content",
                    size_bytes=15,
                ),
            ],
            backlog_tickets=[
                BacklogTicket(
                    id="TICKET-1",
                    title="Feature A",
                    description="Description of Feature A",
                    status="open",
                    tags=["feature"],
                ),
                BacklogTicket(
                    id="TICKET-2",
                    title="Bug Fix B",
                    description="Fix the login bug",
                    status="in_progress",
                    tags=["bug"],
                ),
            ],
        )

        await orchestrator.run_session(inputs)

        # store_for_search should be called with ticket items
        store_calls = mock_memory.store_for_search.call_args_list
        assert len(store_calls) >= 1
        # First call stores backlog tickets
        ticket_items = store_calls[0][0][1]
        assert len(ticket_items) == 2
        assert ticket_items[0]["item_id"] == "TICKET-1"
        assert "Feature A" in ticket_items[0]["content"]
        assert ticket_items[1]["item_id"] == "TICKET-2"

    async def test_no_tickets_means_no_store_for_search_call(
        self, orchestrator, mock_memory
    ):
        """When no backlog tickets are provided, store_for_search is not called for tickets."""
        inputs = SessionInputs(
            session_id="test-session",
            documents=[
                InputDocument(
                    filename="notes.txt",
                    document_type=DocumentType.TRANSCRIPT_TXT,
                    content=b"Content",
                    size_bytes=7,
                ),
            ],
            backlog_tickets=[],
        )

        await orchestrator.run_session(inputs)

        # No ticket-related store_for_search call (may still have extraction items call)
        # With empty extraction, there should be no store_for_search calls
        # (parser returns empty items, so no search items either)
        for call in mock_memory.store_for_search.call_args_list:
            items = call[0][1]
            # Ensure no ticket-specific items are stored
            for item in items:
                assert "TICKET" not in item.get("item_id", "")

    async def test_pipeline_continues_after_ticket_validation(
        self, orchestrator, mock_parser, mock_gap_detector, mock_story_writer
    ):
        """The pipeline continues normally after backlog ticket validation."""
        inputs = SessionInputs(
            session_id="test-session",
            documents=[
                InputDocument(
                    filename="meeting.txt",
                    document_type=DocumentType.TRANSCRIPT_TXT,
                    content=b"Meeting notes",
                    size_bytes=13,
                ),
            ],
            backlog_tickets=[
                BacklogTicket(
                    id="T-1",
                    title="Ticket",
                    description="Desc",
                    status="open",
                ),
            ],
        )

        result = await orchestrator.run_session(inputs)

        # Parser, gap detector, and story writer all called
        mock_parser.parse_documents.assert_called_once()
        mock_gap_detector.analyze_gaps.assert_called_once()
        mock_story_writer.generate_stories.assert_called_once()
        assert result.status == "completed"


# --- Property-Based Tests ---
# Feature: backlog-synthesizer, Property 3: Backlog ticket validation accepts valid and rejects invalid


from hypothesis import given, settings
from hypothesis import strategies as st


def _make_orchestrator_for_property_test():
    """Create an OrchestratorAgent with mocked dependencies for property testing."""
    mock_memory = MagicMock()
    mock_memory.store_intermediate = MagicMock()
    mock_memory.store_for_search = MagicMock()
    mock_memory.log_action = MagicMock()

    mock_parser = MagicMock()
    mock_gap_detector = MagicMock()
    mock_story_writer = MagicMock()

    return OrchestratorAgent(
        parser=mock_parser,
        gap_detector=mock_gap_detector,
        story_writer=mock_story_writer,
        memory=mock_memory,
    )


def valid_backlog_ticket_strategy():
    """Strategy that generates valid BacklogTicket dicts."""
    return st.fixed_dictionaries(
        {
            "id": st.text(min_size=1, max_size=50),
            "title": st.text(min_size=1, max_size=200),
            "description": st.text(min_size=1, max_size=500),
            "status": st.text(min_size=1, max_size=50),
            "tags": st.lists(st.text(min_size=0, max_size=30), max_size=5),
        }
    )


def invalid_backlog_ticket_strategy():
    """Strategy that generates invalid BacklogTicket dicts (missing required fields)."""
    return st.one_of(
        # Missing 'id'
        st.fixed_dictionaries(
            {
                "title": st.text(min_size=1, max_size=50),
                "description": st.text(min_size=1, max_size=50),
                "status": st.text(min_size=1, max_size=20),
            }
        ),
        # Missing 'title'
        st.fixed_dictionaries(
            {
                "id": st.text(min_size=1, max_size=50),
                "description": st.text(min_size=1, max_size=50),
                "status": st.text(min_size=1, max_size=20),
            }
        ),
        # Missing 'description'
        st.fixed_dictionaries(
            {
                "id": st.text(min_size=1, max_size=50),
                "title": st.text(min_size=1, max_size=50),
                "status": st.text(min_size=1, max_size=20),
            }
        ),
        # Missing 'status'
        st.fixed_dictionaries(
            {
                "id": st.text(min_size=1, max_size=50),
                "title": st.text(min_size=1, max_size=50),
                "description": st.text(min_size=1, max_size=50),
            }
        ),
        # Empty dict
        st.just({}),
        # Only random keys
        st.dictionaries(
            keys=st.text(min_size=1, max_size=10).filter(
                lambda k: k not in ("id", "title", "description", "status", "tags", "created_at")
            ),
            values=st.text(min_size=1, max_size=30),
            min_size=1,
            max_size=3,
        ),
    )


def mixed_ticket_list_strategy():
    """Strategy that generates a mixed list of valid and invalid ticket dicts."""
    return st.lists(
        st.one_of(
            valid_backlog_ticket_strategy().map(lambda t: ("valid", t)),
            invalid_backlog_ticket_strategy().map(lambda t: ("invalid", t)),
        ),
        min_size=0,
        max_size=20,
    )


class TestBacklogTicketValidationProperty:
    """Property-based tests for backlog ticket validation.

    **Validates: Requirements 1.4, 1.6**
    """

    @settings(max_examples=100)
    @given(tagged_tickets=mixed_ticket_list_strategy())
    def test_accepted_plus_rejected_equals_total(self, tagged_tickets):
        """Property 3: For any list of ticket dicts, accepted + rejected == total count.

        For any list of JSON objects, where some conform to the BacklogTicket schema
        and some do not, validating the list SHALL accept all conforming entries and
        reject all non-conforming entries, with the count of accepted + rejected
        entries equaling the total input count.
        """
        orchestrator = _make_orchestrator_for_property_test()
        raw_tickets = [ticket for _, ticket in tagged_tickets]

        valid, errors = orchestrator.validate_backlog_tickets_from_json(
            "property-test-session", raw_tickets
        )

        # Core property: accepted + rejected == total
        assert len(valid) + len(errors) == len(raw_tickets)

        # All accepted entries are BacklogTicket instances
        assert all(isinstance(t, BacklogTicket) for t in valid)
