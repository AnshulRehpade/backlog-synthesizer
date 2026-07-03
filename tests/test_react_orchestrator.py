"""Tests for ReAct reasoning integration in the Orchestrator Agent.

Tests each of the 5 decision points:
1. After Parser — empty or few items
2. After Gap Detection — all or mostly duplicates
3. After Story Writer — quality issues
4. On Permanent error
5. Conflicts detected
"""

import json

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from backlog_synthesizer.agents.orchestrator import OrchestratorAgent, SessionResult
from backlog_synthesizer.agents.react_reasoning import ReActReasoner
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
from backlog_synthesizer.tools.errors import PermanentToolError
from backlog_synthesizer.agents.errors import PipelineHaltError


# --- Shared Fixtures ---


@pytest.fixture
def mock_memory():
    """Create a mock MemoryEngine."""
    memory = MagicMock()
    memory.store_intermediate = MagicMock()
    memory.store_for_search = MagicMock()
    memory.log_action = MagicMock()
    return memory


@pytest.fixture
def mock_reasoning_llm():
    """Create a mock LLM tool for ReAct reasoning."""
    llm = MagicMock()
    llm.generate = MagicMock(return_value='{"action": "proceed_normal", "reason": "default", "parameters": {}}')
    return llm


@pytest.fixture
def sample_session_inputs():
    """Create sample SessionInputs."""
    return SessionInputs(
        session_id="test-react-session",
        documents=[
            InputDocument(
                filename="meeting.txt",
                document_type=DocumentType.TRANSCRIPT_TXT,
                content=b"Meeting notes content here",
                size_bytes=26,
            ),
        ],
        backlog_tickets=[],
    )


@pytest.fixture
def sample_extracted_items():
    """Create sample extracted items."""
    return [
        ExtractedItem(
            item_type="feature_request",
            text="Add dark mode support for the dashboard",
            source_chunk_index=0,
            confidence=0.9,
            stakeholder="user",
            tags=["ui"],
        ),
        ExtractedItem(
            item_type="decision",
            text="Use PostgreSQL for the main database",
            source_chunk_index=1,
            confidence=0.85,
            tags=["backend"],
        ),
    ]


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
            tags=["ui"],
        ),
        UserStory(
            title="Database Migration",
            user_story="As a developer, I want PostgreSQL, so that I have reliable storage",
            acceptance_criteria=[
                AcceptanceCriterion(description="Migration script runs without errors"),
                AcceptanceCriterion(description="All existing data preserved"),
            ],
            tags=["backend"],
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
            session_id="test-react-session",
            timestamp=datetime.now(timezone.utc),
        ),
    )


def _make_gap_report(items, classification="new"):
    """Helper to build a GapReport from items with a given classification."""
    entries = [
        GapReportEntry(item=item, classification=classification, confidence=1.0)
        for item in items
    ]
    total_new = sum(1 for e in entries if e.classification == "new")
    total_dup = sum(1 for e in entries if e.classification == "duplicate")
    total_conf = sum(1 for e in entries if e.classification == "conflict")
    total_unproc = sum(1 for e in entries if e.classification == "unprocessed")
    return GapReport(
        entries=entries,
        total_new=total_new,
        total_duplicates=total_dup,
        total_conflicts=total_conf,
        total_unprocessed=total_unproc,
    )


def _create_orchestrator(
    mock_parser, mock_gap_detector, mock_story_writer, mock_memory, mock_reasoning_llm
):
    """Helper to create orchestrator with reasoning LLM."""
    return OrchestratorAgent(
        parser=mock_parser,
        gap_detector=mock_gap_detector,
        story_writer=mock_story_writer,
        memory=mock_memory,
        reasoning_llm=mock_reasoning_llm,
    )


# ===========================================================================
# Decision Point 1: After Parser — empty or few items
# ===========================================================================

class TestDecisionPoint1AfterParser:
    """Tests for empty/few items after parser."""

    @pytest.fixture
    def mock_parser_empty(self):
        """Parser returning empty result."""
        parser = MagicMock()
        parser.parse_documents = AsyncMock(
            return_value=ExtractionResult(items=[], errors=[])
        )
        return parser

    @pytest.fixture
    def mock_parser_few(self):
        """Parser returning only 2 items (less than 3)."""
        items = [
            ExtractedItem(
                item_type="feature_request",
                text="Add feature X",
                source_chunk_index=0,
                confidence=0.9,
                tags=["ui"],
            ),
            ExtractedItem(
                item_type="decision",
                text="Use technology Y",
                source_chunk_index=1,
                confidence=0.85,
                tags=["backend"],
            ),
        ]
        parser = MagicMock()
        parser.parse_documents = AsyncMock(
            return_value=ExtractionResult(items=items, errors=[])
        )
        return parser

    @pytest.fixture
    def mock_gap_detector(self, sample_extracted_items):
        """Default gap detector returning all-new."""
        report = _make_gap_report(sample_extracted_items, "new")
        detector = MagicMock()
        detector.analyze_gaps = AsyncMock(return_value=report)
        return detector

    @pytest.fixture
    def mock_story_writer(self, sample_stories, sample_epics, sample_story_output):
        """Default story writer."""
        writer = MagicMock()
        writer.generate_stories = AsyncMock(return_value=sample_stories)
        writer.group_into_epics = MagicMock(return_value=sample_epics)
        writer.serialize_output = MagicMock(
            return_value=SerializationResult(output=sample_story_output, errors=[])
        )
        return writer

    async def test_empty_items_triggers_reasoning(
        self, mock_parser_empty, mock_gap_detector, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When parser returns 0 items, reasoner is called."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "proceed_empty", "reason": "continue anyway", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser_empty, mock_gap_detector, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        await orchestrator.run_session(sample_session_inputs)
        mock_reasoning_llm.generate.assert_called()

    async def test_halt_action_returns_empty_completed(
        self, mock_parser_empty, mock_gap_detector, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When LLM chooses 'halt', pipeline returns completed with no output."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "halt", "reason": "no items to process", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser_empty, mock_gap_detector, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        assert result.status == "completed"
        assert result.output is None
        assert result.metadata.get("react_note") is not None

    async def test_proceed_action_continues_pipeline(
        self, mock_parser_empty, mock_gap_detector, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When LLM chooses 'proceed_empty', pipeline continues."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "proceed_empty", "reason": "lets go", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser_empty, mock_gap_detector, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        # Gap detector should have been called
        mock_gap_detector.analyze_gaps.assert_called()

    async def test_llm_failure_falls_back_to_default(
        self, mock_parser_empty, mock_gap_detector, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When LLM call fails, fallback to first action (halt) is executed."""
        mock_reasoning_llm.generate = MagicMock(side_effect=RuntimeError("LLM unavailable"))
        orchestrator = _create_orchestrator(
            mock_parser_empty, mock_gap_detector, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        # Default action for empty items is "halt" (first in list)
        assert result.status == "completed"
        assert result.output is None

    async def test_few_items_with_warning(
        self, mock_parser_few, mock_gap_detector, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When parser returns <3 items and LLM says warn, warning is added."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "proceed_with_warning", "reason": "low count", "parameters": {}}'
        )
        # Gap detector needs to handle the 2 items from mock_parser_few
        items = [
            ExtractedItem(
                item_type="feature_request",
                text="Add feature X",
                source_chunk_index=0,
                confidence=0.9,
                tags=["ui"],
            ),
            ExtractedItem(
                item_type="decision",
                text="Use technology Y",
                source_chunk_index=1,
                confidence=0.85,
                tags=["backend"],
            ),
        ]
        report = _make_gap_report(items, "new")
        mock_gap_detector.analyze_gaps = AsyncMock(return_value=report)

        orchestrator = _create_orchestrator(
            mock_parser_few, mock_gap_detector, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        # Should have a react_warning in errors
        assert any(
            e.get("step") == "react_warning" and "Low item count" in e.get("message", "")
            for e in result.errors
        )


# ===========================================================================
# Decision Point 2: After Gap Detection — all or mostly duplicates
# ===========================================================================

class TestDecisionPoint2AfterGapDetection:
    """Tests for all/mostly duplicates after gap detection."""

    @pytest.fixture
    def sample_items(self):
        return [
            ExtractedItem(
                item_type="feature_request",
                text="Add dark mode",
                source_chunk_index=0,
                confidence=0.9,
                tags=["ui"],
            ),
            ExtractedItem(
                item_type="decision",
                text="Use Postgres",
                source_chunk_index=1,
                confidence=0.85,
                tags=["backend"],
            ),
        ]

    @pytest.fixture
    def mock_parser(self, sample_items):
        parser = MagicMock()
        parser.parse_documents = AsyncMock(
            return_value=ExtractionResult(items=sample_items, errors=[])
        )
        return parser

    @pytest.fixture
    def mock_gap_all_duplicates(self, sample_items):
        """Gap detector returning all duplicates."""
        report = _make_gap_report(sample_items, "duplicate")
        detector = MagicMock()
        detector.analyze_gaps = AsyncMock(return_value=report)
        return detector

    @pytest.fixture
    def mock_story_writer(self, sample_stories, sample_epics, sample_story_output):
        writer = MagicMock()
        writer.generate_stories = AsyncMock(return_value=sample_stories)
        writer.group_into_epics = MagicMock(return_value=sample_epics)
        writer.serialize_output = MagicMock(
            return_value=SerializationResult(output=sample_story_output, errors=[])
        )
        return writer

    async def test_all_duplicates_triggers_reasoning(
        self, mock_parser, mock_gap_all_duplicates, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """All-duplicate gap report triggers reasoning."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "proceed_anyway", "reason": "generate stories", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_all_duplicates, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        await orchestrator.run_session(sample_session_inputs)
        mock_reasoning_llm.generate.assert_called()

    async def test_halt_all_duplicates_returns_completed(
        self, mock_parser, mock_gap_all_duplicates, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When LLM chooses halt_all_duplicates, pipeline stops."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "halt_all_duplicates", "reason": "nothing new", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_all_duplicates, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        assert result.status == "completed"
        assert result.output is None
        mock_story_writer.generate_stories.assert_not_called()

    async def test_proceed_anyway_continues(
        self, mock_parser, mock_gap_all_duplicates, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When LLM chooses proceed_anyway, story writer is called."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "proceed_anyway", "reason": "user wants stories", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_all_duplicates, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        await orchestrator.run_session(sample_session_inputs)
        mock_story_writer.generate_stories.assert_called()

    async def test_llm_failure_falls_back_to_default(
        self, mock_parser, mock_gap_all_duplicates, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When LLM fails, default (halt_all_duplicates) is used."""
        mock_reasoning_llm.generate = MagicMock(side_effect=RuntimeError("LLM down"))
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_all_duplicates, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        # Default is first action: halt_all_duplicates
        assert result.status == "completed"
        assert result.output is None


# ===========================================================================
# Decision Point 3: After Story Writer — quality issues
# ===========================================================================

class TestDecisionPoint3AfterStoryWriter:
    """Tests for quality issues after story writer."""

    @pytest.fixture
    def sample_items(self):
        return [
            ExtractedItem(
                item_type="feature_request",
                text="Short text that lacks enough detail for generation",
                source_chunk_index=0,
                confidence=0.3,
                tags=["ui"],
            ),
            ExtractedItem(
                item_type="feature_request",
                text="Another short one insufficient",
                source_chunk_index=1,
                confidence=0.2,
                tags=["ui"],
            ),
            ExtractedItem(
                item_type="feature_request",
                text="Third vague item here",
                source_chunk_index=2,
                confidence=0.4,
                tags=["backend"],
            ),
        ]

    @pytest.fixture
    def refinement_stories(self):
        """Stories where majority need refinement."""
        return [
            UserStory(
                title="Vague Story 1",
                user_story="As a user, I want something, so that things improve",
                acceptance_criteria=[
                    AcceptanceCriterion(description="Criterion 1"),
                    AcceptanceCriterion(description="Criterion 2"),
                ],
                tags=["ui"],
                needs_refinement=True,
            ),
            UserStory(
                title="Vague Story 2",
                user_story="As a user, I want stuff, so that it works",
                acceptance_criteria=[
                    AcceptanceCriterion(description="Criterion 1"),
                    AcceptanceCriterion(description="Criterion 2"),
                ],
                tags=["ui"],
                needs_refinement=True,
            ),
            UserStory(
                title="Good Story",
                user_story="As a user, I want dark mode, so that I can reduce eye strain",
                acceptance_criteria=[
                    AcceptanceCriterion(description="Toggle switch"),
                    AcceptanceCriterion(description="WCAG colors"),
                ],
                tags=["backend"],
                needs_refinement=False,
            ),
        ]

    @pytest.fixture
    def mock_parser(self, sample_items):
        parser = MagicMock()
        parser.parse_documents = AsyncMock(
            return_value=ExtractionResult(items=sample_items, errors=[])
        )
        return parser

    @pytest.fixture
    def mock_gap_detector(self, sample_items):
        report = _make_gap_report(sample_items, "new")
        detector = MagicMock()
        detector.analyze_gaps = AsyncMock(return_value=report)
        return detector

    @pytest.fixture
    def mock_story_writer_refinement(self, refinement_stories, sample_story_output):
        """Story writer that returns stories needing refinement."""
        epics = [Epic(epic_title="Mixed Quality", stories=refinement_stories)]
        output = StoryOutput(
            index=[{"epic_title": "Mixed Quality", "story_count": 3}],
            epics=epics,
            metadata=OutputMetadata(
                session_id="test-react-session",
                timestamp=datetime.now(timezone.utc),
            ),
        )
        writer = MagicMock()
        writer.generate_stories = AsyncMock(return_value=refinement_stories)
        writer.group_into_epics = MagicMock(return_value=epics)
        writer.serialize_output = MagicMock(
            return_value=SerializationResult(output=output, errors=[])
        )
        return writer

    async def test_refinement_triggers_reasoning(
        self, mock_parser, mock_gap_detector, mock_story_writer_refinement,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """Majority needing refinement triggers reasoning."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "return_with_warning", "reason": "quality issues", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_detector, mock_story_writer_refinement,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        assert any(
            "refinement" in e.get("message", "") for e in result.errors
            if e.get("step") == "react_warning"
        )

    async def test_return_normal_no_warning(
        self, mock_parser, mock_gap_detector, mock_story_writer_refinement,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When LLM says return_normal, no quality warning is added."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "return_normal", "reason": "its fine", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_detector, mock_story_writer_refinement,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        assert not any(
            "refinement" in e.get("message", "") for e in result.errors
            if e.get("step") == "react_warning"
        )

    async def test_no_epics_single_epic_action(
        self, mock_parser, mock_gap_detector,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When no epics formed and LLM says return_single_epic, stories grouped."""
        stories = [
            UserStory(
                title="Story A",
                user_story="As a user, I want A, so that B",
                acceptance_criteria=[
                    AcceptanceCriterion(description="C1"),
                    AcceptanceCriterion(description="C2"),
                ],
                tags=["unique1"],
            ),
            UserStory(
                title="Story B",
                user_story="As a user, I want X, so that Y",
                acceptance_criteria=[
                    AcceptanceCriterion(description="C3"),
                    AcceptanceCriterion(description="C4"),
                ],
                tags=["unique2"],
            ),
        ]
        # group_into_epics returns empty list (no shared tags)
        output = StoryOutput(
            index=[],
            epics=[],
            metadata=OutputMetadata(
                session_id="test-react-session",
                timestamp=datetime.now(timezone.utc),
            ),
        )
        single_epic_output = StoryOutput(
            index=[{"epic_title": "General", "story_count": 2}],
            epics=[Epic(epic_title="General", stories=stories)],
            metadata=OutputMetadata(
                session_id="test-react-session",
                timestamp=datetime.now(timezone.utc),
            ),
        )
        writer = MagicMock()
        writer.generate_stories = AsyncMock(return_value=stories)
        writer.group_into_epics = MagicMock(return_value=[])  # No epics!
        writer.serialize_output = MagicMock(
            side_effect=[
                SerializationResult(output=output, errors=[]),
                SerializationResult(output=single_epic_output, errors=[]),
            ]
        )

        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "return_single_epic", "reason": "group them", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_detector, writer,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        # serialize_output should be called twice (second time with single epic)
        assert writer.serialize_output.call_count == 2

    async def test_llm_failure_falls_back_to_default(
        self, mock_parser, mock_gap_detector, mock_story_writer_refinement,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When LLM fails on quality decision, pipeline continues normally."""
        mock_reasoning_llm.generate = MagicMock(side_effect=RuntimeError("LLM error"))
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_detector, mock_story_writer_refinement,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        # Should still produce output despite LLM failure
        assert result.output is not None


# ===========================================================================
# Decision Point 4: On Permanent Error
# ===========================================================================

class TestDecisionPoint4PermanentError:
    """Tests for permanent error handling with ReAct reasoning."""

    @pytest.fixture
    def sample_items(self):
        return [
            ExtractedItem(
                item_type="feature_request",
                text="Add dark mode",
                source_chunk_index=0,
                confidence=0.9,
                tags=["ui"],
            ),
        ]

    @pytest.fixture
    def mock_parser(self, sample_items):
        parser = MagicMock()
        parser.parse_documents = AsyncMock(
            return_value=ExtractionResult(items=sample_items, errors=[])
        )
        return parser

    @pytest.fixture
    def mock_gap_detector_permanent_error(self):
        """Gap detector that raises PermanentToolError."""
        detector = MagicMock()
        detector.analyze_gaps = AsyncMock(
            side_effect=PermanentToolError("Auth failed")
        )
        return detector

    @pytest.fixture
    def mock_story_writer(self, sample_stories, sample_epics, sample_story_output):
        writer = MagicMock()
        writer.generate_stories = AsyncMock(return_value=sample_stories)
        writer.group_into_epics = MagicMock(return_value=sample_epics)
        writer.serialize_output = MagicMock(
            return_value=SerializationResult(output=sample_story_output, errors=[])
        )
        return writer

    async def test_permanent_error_triggers_reasoning(
        self, mock_parser, mock_gap_detector_permanent_error, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """Permanent error triggers ReAct reasoning."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "halt_completely", "reason": "no recovery", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_detector_permanent_error, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        assert result.status == "permanent_failure"
        mock_reasoning_llm.generate.assert_called()

    async def test_return_partial_with_existing_results(
        self, mock_parser, mock_gap_detector_permanent_error, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When LLM says return_partial and partial results exist, returns partial_failure."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "return_partial", "reason": "save what we have", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_detector_permanent_error, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        # Parser completed, so partial results exist
        assert result.status == "partial_failure"

    async def test_halt_completely_returns_permanent_failure(
        self, mock_parser, mock_gap_detector_permanent_error, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When LLM says halt_completely, returns permanent_failure."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "halt_completely", "reason": "cant recover", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_detector_permanent_error, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        assert result.status == "permanent_failure"

    async def test_llm_failure_falls_back_to_halt(
        self, mock_parser, mock_gap_detector_permanent_error, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When LLM fails during permanent error handling, defaults to halt."""
        mock_reasoning_llm.generate = MagicMock(side_effect=RuntimeError("LLM down"))
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_detector_permanent_error, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        # LLM fails -> default is "return_partial" (first action), and extraction_result exists
        # So it returns partial_failure
        assert result.status == "partial_failure"


# ===========================================================================
# Decision Point 5: Conflicts Detected
# ===========================================================================

class TestDecisionPoint5ConflictsDetected:
    """Tests for conflict detection with ReAct reasoning."""

    @pytest.fixture
    def sample_items(self):
        return [
            ExtractedItem(
                item_type="feature_request",
                text="Use MySQL for the database",
                source_chunk_index=0,
                confidence=0.9,
                tags=["backend"],
            ),
            ExtractedItem(
                item_type="feature_request",
                text="Add feature Y",
                source_chunk_index=1,
                confidence=0.8,
                tags=["ui"],
            ),
        ]

    @pytest.fixture
    def mock_parser(self, sample_items):
        parser = MagicMock()
        parser.parse_documents = AsyncMock(
            return_value=ExtractionResult(items=sample_items, errors=[])
        )
        return parser

    @pytest.fixture
    def mock_gap_detector_with_conflicts(self, sample_items):
        """Gap detector with conflicts."""
        entries = [
            GapReportEntry(
                item=sample_items[0],
                classification="conflict",
                confidence=0.7,
                similarity_score=0.65,
                similar_ticket_id="TICKET-42",
            ),
            GapReportEntry(
                item=sample_items[1],
                classification="new",
                confidence=1.0,
            ),
        ]
        report = GapReport(
            entries=entries,
            total_new=1,
            total_duplicates=0,
            total_conflicts=1,
            total_unprocessed=0,
        )
        detector = MagicMock()
        detector.analyze_gaps = AsyncMock(return_value=report)
        return detector

    @pytest.fixture
    def mock_story_writer(self, sample_stories, sample_epics, sample_story_output):
        writer = MagicMock()
        writer.generate_stories = AsyncMock(return_value=sample_stories)
        writer.group_into_epics = MagicMock(return_value=sample_epics)
        writer.serialize_output = MagicMock(
            return_value=SerializationResult(output=sample_story_output, errors=[])
        )
        return writer

    async def test_conflicts_trigger_reasoning(
        self, mock_parser, mock_gap_detector_with_conflicts, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """Conflicts in gap report trigger reasoning."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "proceed_with_conflicts", "reason": "normal flow", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_detector_with_conflicts, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        await orchestrator.run_session(sample_session_inputs)
        # Reasoning was called (at least once for conflicts)
        mock_reasoning_llm.generate.assert_called()

    async def test_add_conflict_summary_adds_metadata(
        self, mock_parser, mock_gap_detector_with_conflicts, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When LLM says add_conflict_summary, conflict details added to errors."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "add_conflict_summary", "reason": "transparency", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_detector_with_conflicts, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        conflict_summaries = [
            e for e in result.errors if e.get("step") == "react_conflict_summary"
        ]
        assert len(conflict_summaries) == 1
        assert "conflicts" in conflict_summaries[0]

    async def test_proceed_with_conflicts_no_summary(
        self, mock_parser, mock_gap_detector_with_conflicts, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When LLM says proceed_with_conflicts, no conflict summary added."""
        mock_reasoning_llm.generate = MagicMock(
            return_value='{"action": "proceed_with_conflicts", "reason": "normal", "parameters": {}}'
        )
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_detector_with_conflicts, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        conflict_summaries = [
            e for e in result.errors if e.get("step") == "react_conflict_summary"
        ]
        assert len(conflict_summaries) == 0

    async def test_llm_failure_falls_back_to_default(
        self, mock_parser, mock_gap_detector_with_conflicts, mock_story_writer,
        mock_memory, mock_reasoning_llm, sample_session_inputs
    ):
        """When LLM fails on conflict decision, continues normally."""
        mock_reasoning_llm.generate = MagicMock(side_effect=RuntimeError("LLM down"))
        orchestrator = _create_orchestrator(
            mock_parser, mock_gap_detector_with_conflicts, mock_story_writer,
            mock_memory, mock_reasoning_llm,
        )
        result = await orchestrator.run_session(sample_session_inputs)
        # Should still produce output
        assert result.output is not None
        # No conflict summary since LLM failed (default is proceed_with_conflicts)
        conflict_summaries = [
            e for e in result.errors if e.get("step") == "react_conflict_summary"
        ]
        assert len(conflict_summaries) == 0


# ===========================================================================
# Test ReActReasoner directly
# ===========================================================================

class TestReActReasoner:
    """Direct unit tests for the ReActReasoner class."""

    async def test_no_llm_returns_default(self):
        """When no LLM tool, returns first available action."""
        reasoner = ReActReasoner(llm_tool=None)
        result = await reasoner.decide(
            "test_point", "test observation",
            [{"action": "first", "description": "d1"}, {"action": "second", "description": "d2"}],
        )
        assert result["action"] == "first"
        assert result["reason"] == "default fallback"

    async def test_valid_json_response(self):
        """When LLM returns valid JSON, parses correctly."""
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(
            return_value='{"action": "second", "reason": "better choice", "parameters": {"key": "val"}}'
        )
        reasoner = ReActReasoner(llm_tool=mock_llm)
        result = await reasoner.decide(
            "test_point", "observation",
            [{"action": "first", "description": "d1"}, {"action": "second", "description": "d2"}],
        )
        assert result["action"] == "second"
        assert result["reason"] == "better choice"
        assert result["parameters"] == {"key": "val"}

    async def test_markdown_fenced_response(self):
        """When LLM wraps response in markdown fences, strips them."""
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(
            return_value='```json\n{"action": "first", "reason": "ok", "parameters": {}}\n```'
        )
        reasoner = ReActReasoner(llm_tool=mock_llm)
        result = await reasoner.decide(
            "test_point", "observation",
            [{"action": "first", "description": "d1"}],
        )
        assert result["action"] == "first"

    async def test_invalid_json_falls_back(self):
        """When LLM returns non-JSON, falls back to first action."""
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="I think you should halt")
        reasoner = ReActReasoner(llm_tool=mock_llm)
        result = await reasoner.decide(
            "test_point", "observation",
            [{"action": "default_action", "description": "d1"}],
        )
        assert result["action"] == "default_action"

    async def test_invalid_action_falls_back(self):
        """When LLM returns action not in available list, falls back."""
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(
            return_value='{"action": "nonexistent", "reason": "?", "parameters": {}}'
        )
        reasoner = ReActReasoner(llm_tool=mock_llm)
        result = await reasoner.decide(
            "test_point", "observation",
            [{"action": "valid_action", "description": "d1"}],
        )
        assert result["action"] == "valid_action"

    async def test_exception_falls_back(self):
        """When LLM raises exception, falls back to first action."""
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(side_effect=Exception("connection error"))
        reasoner = ReActReasoner(llm_tool=mock_llm)
        result = await reasoner.decide(
            "test_point", "observation",
            [{"action": "safe_default", "description": "d1"}],
        )
        assert result["action"] == "safe_default"
