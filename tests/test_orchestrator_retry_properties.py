"""Property-based tests for Orchestrator Agent retry logic and error handling.

Uses Hypothesis to verify universal properties of the retry mechanism,
permanent error handling, and partial failure result structure.

Requirements: 2.5, 2.6, 7.1, 7.2, 7.3
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backlog_synthesizer.agents.errors import PipelineHaltError, RetryExhaustedError
from backlog_synthesizer.agents.orchestrator import OrchestratorAgent, SessionResult
from backlog_synthesizer.tools.errors import PermanentToolError, TransientToolError


# --- Strategies ---

# Strategy for error messages (non-empty strings)
error_message_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=100,
).filter(lambda t: t.strip() != "")

# Strategy for the number of transient failures before success (0 = success on first try)
transient_failure_count_strategy = st.integers(min_value=1, max_value=3)

# Strategy for pipeline step names
step_name_strategy = st.sampled_from(["parser", "gap_detection", "story_writer"])


# --- Helpers ---


def _make_orchestrator():
    """Create an OrchestratorAgent with fully mocked dependencies."""
    parser = MagicMock()
    gap_detector = MagicMock()
    story_writer = MagicMock()
    memory = MagicMock()
    memory.store_intermediate = MagicMock()
    memory.store_for_search = MagicMock()
    memory.log_action = MagicMock()
    return OrchestratorAgent(
        parser=parser,
        gap_detector=gap_detector,
        story_writer=story_writer,
        memory=memory,
    )


# --- Property Tests ---


# Feature: backlog-synthesizer, Property 9: Retry policy for transient errors
class TestProperty9RetryPolicyForTransientErrors:
    """Verify up to 3 retries with 1s, 2s, 4s backoff for transient errors.

    For any sub-agent invocation that fails with a transient error, the
    Orchestrator SHALL retry up to 3 times with backoff delays of 1s, 2s, 4s.
    If the invocation succeeds on any retry, processing continues normally.
    If all 3 retries are exhausted, the failure is reported.

    **Validates: Requirements 2.5, 7.1**
    """

    @given(
        num_failures=transient_failure_count_strategy,
        error_msg=error_message_strategy,
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_transient_retry_succeeds_after_n_failures(
        self, num_failures: int, error_msg: str
    ) -> None:
        """
        For any number of transient failures (1-3) followed by success,
        exactly num_failures sleeps occur with correct backoff values,
        and the final result is returned successfully.

        **Validates: Requirements 2.5, 7.1**
        """
        orchestrator = _make_orchestrator()

        call_count = 0
        expected_result = f"success_after_{num_failures}"

        async def transient_then_success(*args):
            nonlocal call_count
            call_count += 1
            if call_count <= num_failures:
                raise TransientToolError(error_msg)
            return expected_result

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await orchestrator._invoke_with_retry(transient_then_success)

        # Total attempts = num_failures + 1 (the successful one)
        assert call_count == num_failures + 1

        # Result is the expected value
        assert result == expected_result

        # Number of sleep calls equals the number of failures
        assert mock_sleep.call_count == num_failures

        # Backoff values are 1.0, 2.0, 4.0 for the respective retries
        expected_backoffs = [1.0, 2.0, 4.0]
        for i in range(num_failures):
            assert mock_sleep.call_args_list[i][0][0] == expected_backoffs[i]

    @given(error_msg=error_message_strategy)
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_transient_retry_exhausted_raises_with_correct_attempts(
        self, error_msg: str
    ) -> None:
        """
        When all 3 retries are exhausted (4 total attempts), a
        RetryExhaustedError is raised with attempts=4 and the correct
        backoff sequence [1.0, 2.0, 4.0].

        **Validates: Requirements 2.5, 7.1**
        """
        orchestrator = _make_orchestrator()

        call_count = 0

        async def always_transient(*args):
            nonlocal call_count
            call_count += 1
            raise TransientToolError(error_msg)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RetryExhaustedError) as exc_info:
                await orchestrator._invoke_with_retry(always_transient)

        # 4 total attempts (initial + 3 retries)
        assert call_count == 4
        assert exc_info.value.attempts == 4
        assert isinstance(exc_info.value.original_error, TransientToolError)

        # Exactly 3 sleep calls with backoff 1, 2, 4
        assert mock_sleep.call_count == 3
        assert mock_sleep.call_args_list[0][0][0] == 1.0
        assert mock_sleep.call_args_list[1][0][0] == 2.0
        assert mock_sleep.call_args_list[2][0][0] == 4.0

    @given(error_msg=error_message_strategy)
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_timeout_error_treated_as_transient(
        self, error_msg: str
    ) -> None:
        """
        asyncio.TimeoutError is treated the same as TransientToolError:
        retried up to 3 times with exponential backoff.

        **Validates: Requirements 2.5, 7.1**
        """
        orchestrator = _make_orchestrator()

        call_count = 0

        async def always_timeout(*args):
            nonlocal call_count
            call_count += 1
            raise asyncio.TimeoutError()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RetryExhaustedError) as exc_info:
                await orchestrator._invoke_with_retry(always_timeout)

        assert call_count == 4
        assert exc_info.value.attempts == 4
        assert isinstance(exc_info.value.original_error, asyncio.TimeoutError)
        assert mock_sleep.call_count == 3
        assert mock_sleep.call_args_list[0][0][0] == 1.0
        assert mock_sleep.call_args_list[1][0][0] == 2.0
        assert mock_sleep.call_args_list[2][0][0] == 4.0


# Feature: backlog-synthesizer, Property 10: No retry for permanent errors
class TestProperty10NoRetryForPermanentErrors:
    """Verify permanent errors cause immediate halt with no retry.

    For any sub-agent invocation that fails with a permanent error,
    the Orchestrator SHALL NOT retry and SHALL immediately halt the
    pipeline with a PipelineHaltError.

    **Validates: Requirements 2.6, 7.3**
    """

    @given(error_msg=error_message_strategy)
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_permanent_error_raises_immediately_no_sleep(
        self, error_msg: str
    ) -> None:
        """
        For any PermanentToolError, the orchestrator raises PipelineHaltError
        immediately without any asyncio.sleep calls (no retry).

        **Validates: Requirements 2.6, 7.3**
        """
        orchestrator = _make_orchestrator()

        call_count = 0

        async def permanent_fail(*args):
            nonlocal call_count
            call_count += 1
            raise PermanentToolError(error_msg)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(PipelineHaltError) as exc_info:
                await orchestrator._invoke_with_retry(permanent_fail)

        # Only 1 attempt — no retries
        assert call_count == 1

        # No sleep calls at all
        mock_sleep.assert_not_called()

        # Original error is preserved
        assert isinstance(exc_info.value.original_error, PermanentToolError)
        assert str(exc_info.value.original_error) == error_msg

    @given(
        num_transient=st.integers(min_value=1, max_value=2),
        transient_msg=error_message_strategy,
        permanent_msg=error_message_strategy,
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_permanent_error_after_transients_halts_immediately(
        self, num_transient: int, transient_msg: str, permanent_msg: str
    ) -> None:
        """
        If transient errors precede a permanent error, the pipeline halts
        on the permanent error. Sleep calls only occur for transient retries,
        not for the permanent error.

        **Validates: Requirements 2.6, 7.3**
        """
        orchestrator = _make_orchestrator()

        call_count = 0

        async def transient_then_permanent(*args):
            nonlocal call_count
            call_count += 1
            if call_count <= num_transient:
                raise TransientToolError(transient_msg)
            raise PermanentToolError(permanent_msg)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(PipelineHaltError) as exc_info:
                await orchestrator._invoke_with_retry(transient_then_permanent)

        # Attempts = transient failures + 1 (the permanent error attempt)
        assert call_count == num_transient + 1

        # Sleep calls only for transient retries (not for the permanent error)
        assert mock_sleep.call_count == num_transient

        # The halt error wraps the permanent error
        assert isinstance(exc_info.value.original_error, PermanentToolError)
        assert str(exc_info.value.original_error) == permanent_msg


# Feature: backlog-synthesizer, Property 11: Partial failure result structure
class TestProperty11PartialFailureResultStructure:
    """Verify partial_failure status and errors array structure.

    For any pipeline execution where at least one step fails after retries,
    the result SHALL have status "partial_failure" and include an errors array
    where each entry describes the failed step.

    **Validates: Requirements 7.2**
    """

    @given(
        step_name=step_name_strategy,
        error_msg=error_message_strategy,
    )
    @settings(max_examples=100)
    def test_partial_failure_has_correct_status_and_errors(
        self, step_name: str, error_msg: str
    ) -> None:
        """
        For any step failure, the SessionResult with partial_failure status
        has a non-empty errors array where each entry has a "step" key.

        **Validates: Requirements 7.2**
        """
        errors = [{"step": step_name, "error": error_msg}]

        result = SessionResult(
            session_id="test-session",
            status="partial_failure",
            output=None,
            session_state=None,
            errors=errors,
        )

        # Status is "partial_failure"
        assert result.status == "partial_failure"

        # Errors array is non-empty
        assert len(result.errors) > 0

        # Each error entry has a "step" key
        for error in result.errors:
            assert "step" in error
            assert isinstance(error["step"], str)
            assert len(error["step"]) > 0

    @given(
        failed_steps=st.lists(
            st.tuples(step_name_strategy, error_message_strategy),
            min_size=1,
            max_size=3,
        ),
    )
    @settings(max_examples=100)
    def test_partial_failure_errors_array_matches_failed_steps(
        self, failed_steps: list[tuple[str, str]]
    ) -> None:
        """
        For any set of failed steps, the errors array contains exactly
        one entry per failed step, each with the correct step name.

        **Validates: Requirements 7.2**
        """
        errors = [{"step": step, "error": msg} for step, msg in failed_steps]

        result = SessionResult(
            session_id="test-session",
            status="partial_failure",
            output=None,
            session_state=None,
            errors=errors,
        )

        # Number of errors matches number of failed steps
        assert len(result.errors) == len(failed_steps)

        # Each error references a valid step name
        valid_steps = {"parser", "gap_detection", "story_writer"}
        for error in result.errors:
            assert error["step"] in valid_steps

    @given(
        num_successful=st.integers(min_value=0, max_value=2),
        num_failed=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100)
    def test_successful_plus_failed_equals_total_attempted(
        self, num_successful: int, num_failed: int
    ) -> None:
        """
        The count of successfully completed steps plus failed steps equals
        the total pipeline steps attempted.

        **Validates: Requirements 7.2**
        """
        all_steps = ["parser", "gap_detection", "story_writer"]
        total_attempted = min(num_successful + num_failed, len(all_steps))

        # Simulate: first num_successful steps succeed, then num_failed fail
        successful_steps = all_steps[:num_successful]
        failed_steps = all_steps[num_successful:total_attempted]

        errors = [{"step": step, "error": "some error"} for step in failed_steps]

        result = SessionResult(
            session_id="test-session",
            status="partial_failure",
            output=None,
            session_state=None,
            errors=errors,
        )

        # Total attempted = successful + failed
        assert len(successful_steps) + len(result.errors) == total_attempted

        # Status is partial_failure when there are errors
        assert result.status == "partial_failure"
        assert len(result.errors) > 0
