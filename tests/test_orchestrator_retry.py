"""Unit tests for Orchestrator Agent retry logic and timeout enforcement.

Tests verify:
- Retry behavior: transient errors trigger up to 3 retries with correct backoff (1s, 2s, 4s)
- No retry for permanent errors: PermanentToolError causes immediate halt
- Timeout enforcement: calls exceeding 120s are treated as transient failures
- Partial failure: when retries exhausted, result has "partial_failure" status and errors array
- Permanent failure: when permanent error occurs, result has "permanent_failure" status
- Successful retry: if a transient error resolves on retry, pipeline continues

Requirements: 2.5, 2.6, 7.1, 7.2, 7.3, 7.5, 7.6
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backlog_synthesizer.agents.errors import PipelineHaltError, RetryExhaustedError
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
from backlog_synthesizer.tools.errors import PermanentToolError, TransientToolError
from backlog_synthesizer.agents.story_writer import SerializationResult

from datetime import datetime, timezone


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
    """Create sample extracted items."""
    return [
        ExtractedItem(
            item_type="feature_request",
            text="Add dark mode support",
            source_chunk_index=0,
            confidence=0.9,
            stakeholder="user",
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
        total_new=1,
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
    ]


@pytest.fixture
def sample_epics(sample_stories):
    """Create sample Epic objects."""
    return [Epic(epic_title="UI Improvements", stories=sample_stories)]


@pytest.fixture
def sample_story_output(sample_epics):
    """Create a sample StoryOutput."""
    return StoryOutput(
        index=[{"epic_title": "UI Improvements", "story_count": 1}],
        epics=sample_epics,
        metadata=OutputMetadata(
            session_id="test-session-retry",
            timestamp=datetime.now(timezone.utc),
        ),
    )


@pytest.fixture
def sample_session_inputs():
    """Create sample SessionInputs."""
    return SessionInputs(
        session_id="test-session-retry",
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


# --- Test: _invoke_with_retry directly ---


class TestInvokeWithRetry:
    """Tests verifying _invoke_with_retry retry, backoff, and error classification."""

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_successful_first_attempt(self, mock_sleep, orchestrator):
        """A successful first attempt returns immediately without retry."""
        async def success():
            return "result"

        result = await orchestrator._invoke_with_retry(success)

        assert result == "result"
        mock_sleep.assert_not_called()

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_transient_error_retries_with_backoff(self, mock_sleep, orchestrator):
        """Transient errors trigger retries with exponential backoff (1s, 2s, 4s)."""
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientToolError("Server error")
            return "success"

        result = await orchestrator._invoke_with_retry(fail_then_succeed)

        assert result == "success"
        assert call_count == 3
        # Verify backoff: 1s after first failure, 2s after second failure
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_transient_error_exhausts_retries(self, mock_sleep, orchestrator):
        """When all retries exhausted, raises RetryExhaustedError."""
        async def always_fail():
            raise TransientToolError("Server error 503")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await orchestrator._invoke_with_retry(always_fail)

        assert exc_info.value.attempts == 4
        assert isinstance(exc_info.value.original_error, TransientToolError)
        # Verify all 3 backoffs occurred: 1s, 2s, 4s
        assert mock_sleep.call_count == 3
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)
        mock_sleep.assert_any_call(4.0)

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_permanent_error_no_retry(self, mock_sleep, orchestrator):
        """Permanent errors raise PipelineHaltError immediately without retry."""
        async def permanent_fail():
            raise PermanentToolError("HTTP 401 Unauthorized")

        with pytest.raises(PipelineHaltError) as exc_info:
            await orchestrator._invoke_with_retry(permanent_fail)

        assert isinstance(exc_info.value.original_error, PermanentToolError)
        mock_sleep.assert_not_called()

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_timeout_treated_as_transient(self, mock_sleep, orchestrator):
        """asyncio.TimeoutError is treated as transient and triggers retry."""
        call_count = 0

        async def timeout_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()
            return "recovered"

        result = await orchestrator._invoke_with_retry(timeout_then_succeed)

        assert result == "recovered"
        assert call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_timeout_exhausts_retries(self, mock_sleep, orchestrator):
        """TimeoutError on all attempts raises RetryExhaustedError."""
        async def always_timeout():
            raise asyncio.TimeoutError()

        with pytest.raises(RetryExhaustedError) as exc_info:
            await orchestrator._invoke_with_retry(always_timeout)

        assert exc_info.value.attempts == 4
        assert isinstance(exc_info.value.original_error, asyncio.TimeoutError)

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_backoff_sequence_is_correct(self, mock_sleep, orchestrator):
        """Backoff sequence is exactly 1s, 2s, 4s for 3 retries."""
        call_count = 0

        async def fail_all_but_last():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise TransientToolError("Error")
            return "final_success"

        result = await orchestrator._invoke_with_retry(fail_all_but_last)

        assert result == "final_success"
        assert mock_sleep.call_args_list[0][0][0] == 1.0
        assert mock_sleep.call_args_list[1][0][0] == 2.0
        assert mock_sleep.call_args_list[2][0][0] == 4.0

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_permanent_error_after_transient_still_halts(
        self, mock_sleep, orchestrator
    ):
        """If a transient error is followed by a permanent error, pipeline halts."""
        call_count = 0

        async def transient_then_permanent():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TransientToolError("Temporary issue")
            raise PermanentToolError("HTTP 403 Forbidden")

        with pytest.raises(PipelineHaltError) as exc_info:
            await orchestrator._invoke_with_retry(transient_then_permanent)

        assert isinstance(exc_info.value.original_error, PermanentToolError)
        # One sleep for the first transient retry
        mock_sleep.assert_called_once_with(1.0)

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_arguments_passed_to_coroutine(self, mock_sleep, orchestrator):
        """Arguments are correctly forwarded to the coroutine function."""
        async def echo(a, b, c):
            return (a, b, c)

        result = await orchestrator._invoke_with_retry(echo, "x", "y", "z")

        assert result == ("x", "y", "z")


# --- Test: run_session with retry integration ---


class TestRunSessionWithRetry:
    """Tests verifying run_session handles retry errors properly."""

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_permanent_error_in_parser_returns_permanent_failure(
        self, mock_sleep, mock_gap_detector, mock_story_writer, mock_memory,
        sample_session_inputs,
    ):
        """Pipeline returns permanent_failure when parser raises PermanentToolError."""
        parser = MagicMock()
        parser.parse_documents = AsyncMock(
            side_effect=PermanentToolError("HTTP 401 Unauthorized")
        )

        orchestrator = OrchestratorAgent(
            parser=parser,
            gap_detector=mock_gap_detector,
            story_writer=mock_story_writer,
            memory=mock_memory,
        )

        result = await orchestrator.run_session(sample_session_inputs)

        assert result.status == "permanent_failure"
        assert any("parser" in e.get("step", "") for e in result.errors)
        mock_gap_detector.analyze_gaps.assert_not_called()
        mock_story_writer.generate_stories.assert_not_called()

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_transient_error_in_parser_retries_exhausted(
        self, mock_sleep, mock_gap_detector, mock_story_writer, mock_memory,
        sample_session_inputs,
    ):
        """Pipeline returns partial_failure when parser retries exhausted."""
        parser = MagicMock()
        parser.parse_documents = AsyncMock(
            side_effect=TransientToolError("HTTP 503 Service Unavailable")
        )

        orchestrator = OrchestratorAgent(
            parser=parser,
            gap_detector=mock_gap_detector,
            story_writer=mock_story_writer,
            memory=mock_memory,
        )

        result = await orchestrator.run_session(sample_session_inputs)

        assert result.status == "partial_failure"
        assert any("parser" in e.get("step", "") for e in result.errors)
        # Verify it attempted retries (sleep called 3 times)
        assert mock_sleep.call_count == 3

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_transient_error_resolved_on_retry_continues_pipeline(
        self, mock_sleep, mock_gap_detector, mock_story_writer, mock_memory,
        sample_session_inputs, sample_extraction_result,
    ):
        """Pipeline continues if transient error resolves on retry."""
        call_count = 0

        async def fail_once_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TransientToolError("HTTP 429 Too Many Requests")
            return sample_extraction_result

        parser = MagicMock()
        parser.parse_documents = AsyncMock(side_effect=fail_once_then_succeed)

        orchestrator = OrchestratorAgent(
            parser=parser,
            gap_detector=mock_gap_detector,
            story_writer=mock_story_writer,
            memory=mock_memory,
        )

        result = await orchestrator.run_session(sample_session_inputs)

        assert result.status == "completed"
        mock_gap_detector.analyze_gaps.assert_called_once()
        mock_story_writer.generate_stories.assert_called_once()

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_permanent_error_in_gap_detection_returns_permanent_failure(
        self, mock_sleep, mock_parser, mock_story_writer, mock_memory,
        sample_session_inputs,
    ):
        """Pipeline returns permanent_failure when gap detection raises PermanentToolError."""
        gap_detector = MagicMock()
        gap_detector.analyze_gaps = AsyncMock(
            side_effect=PermanentToolError("HTTP 403 Forbidden")
        )

        orchestrator = OrchestratorAgent(
            parser=mock_parser,
            gap_detector=gap_detector,
            story_writer=mock_story_writer,
            memory=mock_memory,
        )

        result = await orchestrator.run_session(sample_session_inputs)

        assert result.status == "permanent_failure"
        assert any("gap_detection" in e.get("step", "") for e in result.errors)
        mock_story_writer.generate_stories.assert_not_called()

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_permanent_error_in_story_writer_returns_permanent_failure(
        self, mock_sleep, mock_parser, mock_gap_detector, mock_memory,
        sample_session_inputs,
    ):
        """Pipeline returns permanent_failure when story writer raises PermanentToolError."""
        story_writer = MagicMock()
        story_writer.generate_stories = AsyncMock(
            side_effect=PermanentToolError("HTTP 404 Not Found")
        )

        orchestrator = OrchestratorAgent(
            parser=mock_parser,
            gap_detector=mock_gap_detector,
            story_writer=story_writer,
            memory=mock_memory,
        )

        result = await orchestrator.run_session(sample_session_inputs)

        assert result.status == "permanent_failure"
        assert any("story_writer" in e.get("step", "") for e in result.errors)

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_transient_error_in_gap_detection_retries_exhausted(
        self, mock_sleep, mock_parser, mock_story_writer, mock_memory,
        sample_session_inputs,
    ):
        """Pipeline returns partial_failure when gap detection retries exhausted."""
        gap_detector = MagicMock()
        gap_detector.analyze_gaps = AsyncMock(
            side_effect=TransientToolError("HTTP 500 Internal Server Error")
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
        mock_story_writer.generate_stories.assert_not_called()

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_session_state_updated_with_permanent_failure(
        self, mock_sleep, mock_gap_detector, mock_story_writer, mock_memory,
        sample_session_inputs,
    ):
        """Session state is updated to permanent_failure when pipeline halts."""
        parser = MagicMock()
        parser.parse_documents = AsyncMock(
            side_effect=PermanentToolError("Auth failure")
        )

        orchestrator = OrchestratorAgent(
            parser=parser,
            gap_detector=mock_gap_detector,
            story_writer=mock_story_writer,
            memory=mock_memory,
        )

        await orchestrator.run_session(sample_session_inputs)

        # Last store_intermediate for session_state should have permanent_failure
        last_session_call = None
        for call in mock_memory.store_intermediate.call_args_list:
            if call[0][1] == "session_state":
                last_session_call = call
        assert last_session_call is not None
        stored = last_session_call[0][2]
        assert stored["status"] == "permanent_failure"

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_general_exception_still_returns_partial_failure(
        self, mock_sleep, mock_gap_detector, mock_story_writer, mock_memory,
        sample_session_inputs,
    ):
        """Non-tool exceptions still result in partial_failure (backwards compat)."""
        parser = MagicMock()
        parser.parse_documents = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        orchestrator = OrchestratorAgent(
            parser=parser,
            gap_detector=mock_gap_detector,
            story_writer=mock_story_writer,
            memory=mock_memory,
        )

        result = await orchestrator.run_session(sample_session_inputs)

        assert result.status == "partial_failure"
        assert any("parser" in e.get("step", "") for e in result.errors)
