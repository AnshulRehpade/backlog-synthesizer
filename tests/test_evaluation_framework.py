"""Tests for the evaluation framework pipeline.

Tests the EvaluationFramework class including keyword overlap computation,
LLM judge scoring, pipeline execution, and failure handling.

Requirements: 8.2, 8.3, 8.4, 8.5, 8.6
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backlog_synthesizer.evaluation.framework import EvaluationFramework
from backlog_synthesizer.models.evaluation import (
    EvaluationCaseResult,
    EvaluationReport,
    GoldenEntry,
    JudgeScores,
)
from backlog_synthesizer.models.output import (
    AcceptanceCriterion,
    Epic,
    OutputMetadata,
    StoryOutput,
    UserStory,
)
from datetime import datetime, timezone


# --- Helpers ---


def make_user_story(
    title: str = "Test Story",
    criteria_texts: list[str] | None = None,
    tags: list[str] | None = None,
) -> UserStory:
    """Create a UserStory with given criteria texts."""
    if criteria_texts is None:
        criteria_texts = ["User can login", "User can logout"]
    if tags is None:
        tags = ["auth"]
    return UserStory(
        title=title,
        user_story="As a user, I want to login, so that I can access my account",
        acceptance_criteria=[
            AcceptanceCriterion(description=text) for text in criteria_texts
        ],
        tags=tags,
    )


def make_story_output(stories: list[UserStory] | None = None) -> StoryOutput:
    """Create a StoryOutput wrapping given stories in a single epic."""
    if stories is None:
        stories = [make_user_story()]
    return StoryOutput(
        index=[{"epic_title": "Test Epic", "story_count": len(stories)}],
        epics=[Epic(epic_title="Test Epic", stories=stories)],
        metadata=OutputMetadata(
            session_id="eval-test",
            timestamp=datetime.now(tz=timezone.utc),
        ),
    )


class FakeSessionResult:
    """Fake SessionResult for testing."""

    def __init__(self, output=None, status="completed", errors=None):
        self.session_id = "eval-test-session"
        self.status = status
        self.output = output
        self.errors = errors or []


class FakePipeline:
    """Fake OrchestratorAgent for testing."""

    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.calls = []

    async def run_session(self, inputs):
        self.calls.append(inputs)
        if self._raises:
            raise self._raises
        return self._result


class FakeLLMJudge:
    """Fake LLMGenerationTool that returns predetermined scores."""

    def __init__(self, response: str = '{"relevance": 4, "completeness": 3, "clarity": 5}'):
        self._response = response
        self.calls = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append((prompt, system_prompt))
        return self._response


# --- Keyword Overlap Tests ---


class TestComputeKeywordOverlap:
    """Tests for _compute_keyword_overlap method."""

    def _make_framework(self):
        return EvaluationFramework(
            golden_dataset=[],
            pipeline=FakePipeline(),
            llm_judge=FakeLLMJudge(),
        )

    def test_empty_expected_returns_1(self):
        """If expected is empty, score should be 1.0."""
        fw = self._make_framework()
        score = fw._compute_keyword_overlap(["hello", "world"], [])
        assert score == 1.0

    def test_empty_generated_returns_0(self):
        """If generated is empty but expected is not, score should be 0.0."""
        fw = self._make_framework()
        score = fw._compute_keyword_overlap([], ["hello", "world"])
        assert score == 0.0

    def test_identical_lists_returns_1(self):
        """Identical token lists should produce score 1.0."""
        fw = self._make_framework()
        tokens = ["user", "can", "login"]
        score = fw._compute_keyword_overlap(tokens, tokens)
        assert score == 1.0

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        fw = self._make_framework()
        score = fw._compute_keyword_overlap(["Hello", "WORLD"], ["hello", "world"])
        assert score == 1.0

    def test_partial_overlap(self):
        """Partial overlap should produce a proportional score."""
        fw = self._make_framework()
        score = fw._compute_keyword_overlap(
            ["user", "login"], ["user", "login", "logout", "password"]
        )
        assert score == 0.5

    def test_no_overlap(self):
        """No matching tokens should produce score 0.0."""
        fw = self._make_framework()
        score = fw._compute_keyword_overlap(["apple", "banana"], ["car", "house"])
        assert score == 0.0

    def test_both_empty(self):
        """Both empty should return 1.0 (empty expected)."""
        fw = self._make_framework()
        score = fw._compute_keyword_overlap([], [])
        assert score == 1.0


# --- Property-Based Tests: Keyword Overlap ---


# Feature: backlog-synthesizer, Property 12: Keyword overlap score computation
class TestComputeKeywordOverlapProperty:
    """Property-based tests for _compute_keyword_overlap method.

    **Validates: Requirements 8.3**
    """

    def _make_framework(self):
        return EvaluationFramework(
            golden_dataset=[],
            pipeline=FakePipeline(),
            llm_judge=FakeLLMJudge(),
        )

    @given(
        generated=st.lists(st.text(min_size=1, max_size=20), max_size=30),
        expected=st.lists(st.text(min_size=1, max_size=20), max_size=30),
    )
    @settings(max_examples=100)
    def test_score_always_in_unit_interval(self, generated: list[str], expected: list[str]):
        """The keyword overlap score SHALL always be in [0.0, 1.0]."""
        fw = self._make_framework()
        score = fw._compute_keyword_overlap(generated, expected)
        assert 0.0 <= score <= 1.0

    @given(
        generated=st.lists(st.text(min_size=1, max_size=20), max_size=30),
        expected=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=30),
    )
    @settings(max_examples=100)
    def test_score_equals_matching_count_over_expected_length(
        self, generated: list[str], expected: list[str]
    ):
        """The score SHALL equal matching token count / len(expected), case-insensitive."""
        fw = self._make_framework()
        score = fw._compute_keyword_overlap(generated, expected)

        # Independently compute the expected score
        generated_lower = {token.lower() for token in generated}
        expected_lower = [token.lower() for token in expected]
        matches = sum(1 for token in expected_lower if token in generated_lower)
        expected_score = matches / len(expected_lower)

        assert abs(score - expected_score) < 1e-9

    @given(generated=st.lists(st.text(min_size=0, max_size=20), max_size=30))
    @settings(max_examples=100)
    def test_empty_expected_returns_one(self, generated: list[str]):
        """If expected is empty, the score SHALL be 1.0."""
        fw = self._make_framework()
        score = fw._compute_keyword_overlap(generated, [])
        assert score == 1.0


# --- LLM Judge Tests ---


class TestLLMJudgeScore:
    """Tests for _llm_judge_score method."""

    @pytest.mark.asyncio
    async def test_valid_response_parsed(self):
        """LLM judge with valid JSON response should parse correctly."""
        judge = FakeLLMJudge('{"relevance": 4, "completeness": 3, "clarity": 5}')
        fw = EvaluationFramework(
            golden_dataset=[],
            pipeline=FakePipeline(),
            llm_judge=judge,
        )
        story = make_user_story()
        expected = make_user_story()
        scores = await fw._llm_judge_score(story, expected)
        assert scores.relevance == 4
        assert scores.completeness == 3
        assert scores.clarity == 5

    @pytest.mark.asyncio
    async def test_invalid_response_falls_back(self):
        """Invalid LLM response should fall back to scores of 1."""
        judge = FakeLLMJudge("not valid json")
        fw = EvaluationFramework(
            golden_dataset=[],
            pipeline=FakePipeline(),
            llm_judge=judge,
        )
        story = make_user_story()
        expected = make_user_story()
        scores = await fw._llm_judge_score(story, expected)
        assert scores.relevance == 1
        assert scores.completeness == 1
        assert scores.clarity == 1

    @pytest.mark.asyncio
    async def test_markdown_code_block_response(self):
        """LLM response wrapped in markdown code blocks should be parsed."""
        judge = FakeLLMJudge('```json\n{"relevance": 5, "completeness": 4, "clarity": 3}\n```')
        fw = EvaluationFramework(
            golden_dataset=[],
            pipeline=FakePipeline(),
            llm_judge=judge,
        )
        story = make_user_story()
        expected = make_user_story()
        scores = await fw._llm_judge_score(story, expected)
        assert scores.relevance == 5
        assert scores.completeness == 4
        assert scores.clarity == 3


# --- Run Evaluation Tests ---


class TestRunEvaluation:
    """Tests for the full run_evaluation pipeline."""

    @pytest.mark.asyncio
    async def test_successful_evaluation(self):
        """Successful pipeline run should produce proper scores."""
        stories = [make_user_story(criteria_texts=["user can login", "user can logout"])]
        output = make_story_output(stories)
        pipeline = FakePipeline(result=FakeSessionResult(output=output))
        judge = FakeLLMJudge('{"relevance": 4, "completeness": 4, "clarity": 4}')

        golden = [
            GoldenEntry(
                transcript="We need login functionality",
                expected_stories=[
                    make_user_story(criteria_texts=["user can login", "user can logout"])
                ],
            )
        ]

        fw = EvaluationFramework(
            golden_dataset=golden,
            pipeline=pipeline,
            llm_judge=judge,
        )

        report = await fw.run_evaluation()

        assert len(report.results) == 1
        assert report.results[0].failure_reason is None
        assert report.results[0].keyword_overlap_score == 1.0
        assert report.aggregate_keyword_overlap_mean == 1.0
        assert report.aggregate_keyword_overlap_min == 1.0

    @pytest.mark.asyncio
    async def test_pipeline_failure_records_reason(self):
        """Pipeline exception should record failure with score 0."""
        pipeline = FakePipeline(raises=RuntimeError("LLM API timeout"))
        judge = FakeLLMJudge()

        golden = [
            GoldenEntry(
                transcript="Some transcript",
                expected_stories=[make_user_story()],
            )
        ]

        fw = EvaluationFramework(
            golden_dataset=golden,
            pipeline=pipeline,
            llm_judge=judge,
        )

        report = await fw.run_evaluation()

        assert len(report.results) == 1
        assert report.results[0].keyword_overlap_score == 0.0
        assert report.results[0].failure_reason is not None
        assert "LLM API timeout" in report.results[0].failure_reason
        assert report.aggregate_keyword_overlap_mean == 0.0

    @pytest.mark.asyncio
    async def test_pipeline_no_output_records_failure(self):
        """Pipeline returning no output should record failure."""
        pipeline = FakePipeline(
            result=FakeSessionResult(output=None, status="permanent_failure")
        )
        judge = FakeLLMJudge()

        golden = [
            GoldenEntry(
                transcript="Some transcript",
                expected_stories=[make_user_story()],
            )
        ]

        fw = EvaluationFramework(
            golden_dataset=golden,
            pipeline=pipeline,
            llm_judge=judge,
        )

        report = await fw.run_evaluation()

        assert len(report.results) == 1
        assert report.results[0].keyword_overlap_score == 0.0
        assert report.results[0].failure_reason is not None
        assert "permanent_failure" in report.results[0].failure_reason

    @pytest.mark.asyncio
    async def test_multiple_entries_continues_on_failure(self):
        """Framework should continue with remaining entries after failure."""
        stories = [make_user_story(criteria_texts=["user can login", "system validates"])]
        output = make_story_output(stories)

        # First call fails, second succeeds
        call_count = 0

        class AlternatingPipeline:
            async def run_session(self, inputs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("First call fails")
                return FakeSessionResult(output=output)

        judge = FakeLLMJudge('{"relevance": 3, "completeness": 3, "clarity": 3}')

        golden = [
            GoldenEntry(
                transcript="First transcript",
                expected_stories=[make_user_story()],
            ),
            GoldenEntry(
                transcript="Second transcript",
                expected_stories=[
                    make_user_story(criteria_texts=["user can login", "system validates"])
                ],
            ),
        ]

        fw = EvaluationFramework(
            golden_dataset=golden,
            pipeline=AlternatingPipeline(),
            llm_judge=judge,
        )

        report = await fw.run_evaluation()

        assert len(report.results) == 2
        # First entry failed
        assert report.results[0].failure_reason is not None
        assert report.results[0].keyword_overlap_score == 0.0
        # Second entry succeeded
        assert report.results[1].failure_reason is None
        assert report.results[1].keyword_overlap_score == 1.0

    @pytest.mark.asyncio
    async def test_aggregate_metrics_computed_correctly(self):
        """Aggregate metrics should reflect mean and min correctly."""
        # Two entries with different overlap scores
        # story1 has tokens: "alpha", "beta" (from criteria1) + "filler", "text" (from criteria2)
        # expected has tokens: "alpha", "beta", "gamma", "delta" (from criteria1) + "filler", "text" (from criteria2)
        # story1 matches 4/6 expected tokens = 0.667
        # story2 has tokens that exactly match expected = 6/6 = 1.0
        story1 = [make_user_story(criteria_texts=["alpha beta", "filler text"])]
        story2 = [make_user_story(criteria_texts=["alpha beta gamma delta", "filler text"])]

        class MultiPipeline:
            def __init__(self):
                self._call_count = 0

            async def run_session(self, inputs):
                self._call_count += 1
                if self._call_count == 1:
                    return FakeSessionResult(output=make_story_output(story1))
                return FakeSessionResult(output=make_story_output(story2))

        judge = FakeLLMJudge('{"relevance": 4, "completeness": 3, "clarity": 5}')

        golden = [
            GoldenEntry(
                transcript="First",
                expected_stories=[make_user_story(criteria_texts=["alpha beta gamma delta", "filler text"])],
            ),
            GoldenEntry(
                transcript="Second",
                expected_stories=[make_user_story(criteria_texts=["alpha beta gamma delta", "filler text"])],
            ),
        ]

        fw = EvaluationFramework(
            golden_dataset=golden,
            pipeline=MultiPipeline(),
            llm_judge=judge,
        )

        report = await fw.run_evaluation()

        assert len(report.results) == 2
        # story1 generated tokens: {alpha, beta, filler, text} vs expected: [alpha, beta, gamma, delta, filler, text]
        # Matches: alpha, beta, filler, text = 4/6
        assert abs(report.results[0].keyword_overlap_score - 4.0 / 6.0) < 1e-9
        # story2 generated tokens: {alpha, beta, gamma, delta, filler, text} vs expected: [alpha, beta, gamma, delta, filler, text]
        # Matches: all 6 = 6/6 = 1.0
        assert report.results[1].keyword_overlap_score == 1.0
        expected_mean = (4.0 / 6.0 + 1.0) / 2.0
        assert abs(report.aggregate_keyword_overlap_mean - expected_mean) < 1e-9
        assert abs(report.aggregate_keyword_overlap_min - 4.0 / 6.0) < 1e-9

    @pytest.mark.asyncio
    async def test_empty_golden_dataset(self):
        """Empty golden dataset should produce empty report."""
        fw = EvaluationFramework(
            golden_dataset=[],
            pipeline=FakePipeline(),
            llm_judge=FakeLLMJudge(),
        )

        report = await fw.run_evaluation()

        assert len(report.results) == 0
        assert report.aggregate_keyword_overlap_mean == 0.0
        assert report.aggregate_keyword_overlap_min == 0.0
