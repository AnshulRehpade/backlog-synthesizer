"""Evaluation pipeline for the Backlog Synthesizer system.

Executes the full pipeline against a golden dataset and computes quality
metrics using keyword overlap and LLM-as-judge scoring.

Requirements: 8.2, 8.3, 8.4, 8.5, 8.6
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

from backlog_synthesizer.models.evaluation import (
    EvaluationCaseResult,
    EvaluationReport,
    GoldenEntry,
    JudgeScores,
)
from backlog_synthesizer.models.inputs import (
    DocumentType,
    InputDocument,
    SessionInputs,
)
from backlog_synthesizer.models.output import UserStory

if TYPE_CHECKING:
    from backlog_synthesizer.agents.orchestrator import OrchestratorAgent
    from backlog_synthesizer.tools.interfaces import LLMGenerationTool

logger = logging.getLogger(__name__)


class EvaluationFramework:
    """Runs the Backlog Synthesizer pipeline against a golden dataset and scores outputs.

    Computes two types of metrics:
    - Keyword overlap: normalized case-insensitive token matching between
      generated and expected acceptance criteria.
    - LLM-as-judge: an independent LLM scores each generated story on
      relevance, completeness, and clarity (1-5 scale).

    If the pipeline fails for a given entry, the framework records the failure
    reason, assigns a score of 0, and continues with remaining entries.

    Requirements: 8.2, 8.3, 8.4, 8.5, 8.6
    """

    def __init__(
        self,
        golden_dataset: list[GoldenEntry],
        pipeline: OrchestratorAgent,
        llm_judge: LLMGenerationTool,
    ) -> None:
        """Initialize the evaluation framework.

        Args:
            golden_dataset: List of golden entries with transcripts and expected stories.
            pipeline: The OrchestratorAgent instance to run the pipeline.
            llm_judge: An LLMGenerationTool instance used for LLM-as-judge scoring.
        """
        self._golden_dataset = golden_dataset
        self._pipeline = pipeline
        self._llm_judge = llm_judge

    async def run_evaluation(self) -> EvaluationReport:
        """Execute the full evaluation pipeline against each golden entry.

        For each entry in the golden dataset:
        1. Converts the transcript into SessionInputs.
        2. Runs the pipeline via OrchestratorAgent.run_session.
        3. Compares generated stories to expected stories using keyword overlap.
        4. Scores generated stories using the LLM-as-judge.

        If the pipeline fails for any entry, records the failure reason, assigns
        score 0 for all metrics, and continues with remaining entries.

        Returns:
            An EvaluationReport with per-case results and aggregate metrics
            (mean and minimum for keyword overlap; mean for judge dimensions).
        """
        results: list[EvaluationCaseResult] = []

        for idx, entry in enumerate(self._golden_dataset):
            try:
                result = await self._evaluate_single_entry(idx, entry)
            except Exception as e:
                logger.error(
                    "Pipeline failed for golden entry %d: %s", idx, str(e)
                )
                result = EvaluationCaseResult(
                    case_index=idx,
                    keyword_overlap_score=0.0,
                    judge_scores=None,
                    failure_reason=str(e),
                )
            results.append(result)

        return self._build_report(results)

    async def _evaluate_single_entry(
        self, case_index: int, entry: GoldenEntry
    ) -> EvaluationCaseResult:
        """Evaluate a single golden dataset entry through the pipeline.

        Args:
            case_index: The index of this entry in the golden dataset.
            entry: The golden entry containing transcript and expected stories.

        Returns:
            An EvaluationCaseResult with computed scores.

        Raises:
            Exception: If the pipeline fails to produce output.
        """
        # Convert transcript to SessionInputs
        session_inputs = self._create_session_inputs(entry.transcript)

        # Run the pipeline
        session_result = await self._pipeline.run_session(session_inputs)

        # Check if pipeline produced output
        if session_result.output is None:
            failure_reason = (
                f"Pipeline returned status '{session_result.status}' "
                f"with no output. Errors: {session_result.errors}"
            )
            return EvaluationCaseResult(
                case_index=case_index,
                keyword_overlap_score=0.0,
                judge_scores=None,
                failure_reason=failure_reason,
            )

        # Extract generated stories from pipeline output
        generated_stories: list[UserStory] = []
        for epic in session_result.output.epics:
            generated_stories.extend(epic.stories)

        # Compute keyword overlap across all stories
        generated_keywords = self._extract_keywords_from_stories(generated_stories)
        expected_keywords = self._extract_keywords_from_stories(entry.expected_stories)
        keyword_score = self._compute_keyword_overlap(
            generated_keywords, expected_keywords
        )

        # Compute LLM-as-judge scores (average across story pairs)
        judge_scores = await self._compute_aggregate_judge_scores(
            generated_stories, entry.expected_stories
        )

        return EvaluationCaseResult(
            case_index=case_index,
            keyword_overlap_score=keyword_score,
            judge_scores=judge_scores,
        )

    def _compute_keyword_overlap(
        self, generated: list[str], expected: list[str]
    ) -> float:
        """Compute normalized keyword overlap score between generated and expected tokens.

        The score is the number of case-insensitive matching tokens found in both
        generated and expected lists, divided by the total token count in the
        expected list.

        Args:
            generated: List of keyword tokens from generated stories.
            expected: List of keyword tokens from expected stories.

        Returns:
            A float score between 0.0 and 1.0. Returns 1.0 if expected is empty.
        """
        if not expected:
            return 1.0

        # Lowercase all tokens for case-insensitive comparison
        generated_lower = {token.lower() for token in generated}
        expected_lower = [token.lower() for token in expected]

        # Count matches: how many expected tokens appear in the generated set
        matches = sum(1 for token in expected_lower if token in generated_lower)

        return matches / len(expected_lower)

    async def _llm_judge_score(
        self, story: UserStory, expected: UserStory
    ) -> JudgeScores:
        """Score a generated story against the expected story using LLM-as-judge.

        Asks the LLM to independently evaluate the generated story on three
        dimensions: relevance, completeness, and clarity, each on a 1-5 scale.

        Args:
            story: The generated UserStory to evaluate.
            expected: The expected (golden) UserStory for comparison.

        Returns:
            JudgeScores with integer ratings for relevance, completeness, clarity.
        """
        prompt = self._build_judge_prompt(story, expected)
        system_prompt = (
            "You are an expert evaluator of user stories. "
            "Score the generated story compared to the expected story on three dimensions:\n"
            "- Relevance (1-5): How well does the generated story address the same topic/need?\n"
            "- Completeness (1-5): How fully does the generated story cover the expected criteria?\n"
            "- Clarity (1-5): How clear and well-structured is the generated story?\n\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            '{"relevance": <1-5>, "completeness": <1-5>, "clarity": <1-5>}'
        )

        response = self._llm_judge.generate(prompt=prompt, system_prompt=system_prompt)

        return self._parse_judge_response(response)

    def _build_judge_prompt(self, story: UserStory, expected: UserStory) -> str:
        """Build the evaluation prompt for the LLM judge.

        Args:
            story: The generated story.
            expected: The expected story.

        Returns:
            A formatted prompt string.
        """
        generated_criteria = "\n".join(
            f"  - {ac.description}" for ac in story.acceptance_criteria
        )
        expected_criteria = "\n".join(
            f"  - {ac.description}" for ac in expected.acceptance_criteria
        )

        return (
            f"## Generated Story\n"
            f"Title: {story.title}\n"
            f"User Story: {story.user_story}\n"
            f"Acceptance Criteria:\n{generated_criteria}\n"
            f"Tags: {', '.join(story.tags)}\n\n"
            f"## Expected Story\n"
            f"Title: {expected.title}\n"
            f"User Story: {expected.user_story}\n"
            f"Acceptance Criteria:\n{expected_criteria}\n"
            f"Tags: {', '.join(expected.tags)}\n\n"
            f"Score the generated story against the expected story."
        )

    def _parse_judge_response(self, response: str) -> JudgeScores:
        """Parse the LLM judge response into JudgeScores.

        Attempts to extract a JSON object from the response. Falls back to
        default scores of 1 if parsing fails.

        Args:
            response: The raw LLM response text.

        Returns:
            A JudgeScores instance.
        """
        try:
            # Try to find JSON in the response
            response_stripped = response.strip()
            # Handle case where response might have markdown code blocks
            if "```" in response_stripped:
                json_start = response_stripped.find("{")
                json_end = response_stripped.rfind("}") + 1
                response_stripped = response_stripped[json_start:json_end]

            data = json.loads(response_stripped)
            return JudgeScores(
                relevance=max(1, min(5, int(data.get("relevance", 1)))),
                completeness=max(1, min(5, int(data.get("completeness", 1)))),
                clarity=max(1, min(5, int(data.get("clarity", 1)))),
            )
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logger.warning("Failed to parse LLM judge response: %s. Response: %s", e, response)
            return JudgeScores(relevance=1, completeness=1, clarity=1)

    async def _compute_aggregate_judge_scores(
        self, generated: list[UserStory], expected: list[UserStory]
    ) -> JudgeScores | None:
        """Compute aggregate LLM judge scores across story pairs.

        Pairs generated stories with expected stories (by index) and averages
        the scores. If there are more generated stories than expected, extra
        stories are scored against the last expected story.

        Args:
            generated: List of generated user stories.
            expected: List of expected user stories.

        Returns:
            Averaged JudgeScores, or None if no stories could be scored.
        """
        if not generated or not expected:
            return None

        all_scores: list[JudgeScores] = []

        for i, gen_story in enumerate(generated):
            # Pair with corresponding expected story, or last one if fewer expected
            exp_idx = min(i, len(expected) - 1)
            try:
                scores = await self._llm_judge_score(gen_story, expected[exp_idx])
                all_scores.append(scores)
            except Exception as e:
                logger.warning(
                    "LLM judge scoring failed for story %d: %s", i, str(e)
                )

        if not all_scores:
            return None

        avg_relevance = round(
            sum(s.relevance for s in all_scores) / len(all_scores)
        )
        avg_completeness = round(
            sum(s.completeness for s in all_scores) / len(all_scores)
        )
        avg_clarity = round(
            sum(s.clarity for s in all_scores) / len(all_scores)
        )

        return JudgeScores(
            relevance=max(1, min(5, avg_relevance)),
            completeness=max(1, min(5, avg_completeness)),
            clarity=max(1, min(5, avg_clarity)),
        )

    def _extract_keywords_from_stories(self, stories: list[UserStory]) -> list[str]:
        """Extract keyword tokens from the acceptance criteria of a list of stories.

        Splits acceptance criteria descriptions into whitespace-separated tokens.

        Args:
            stories: List of UserStory objects.

        Returns:
            A flat list of all tokens from acceptance criteria text.
        """
        tokens: list[str] = []
        for story in stories:
            for criterion in story.acceptance_criteria:
                tokens.extend(criterion.description.split())
        return tokens

    def _create_session_inputs(self, transcript: str) -> SessionInputs:
        """Convert a transcript string into SessionInputs for the pipeline.

        Creates a SessionInputs object with the transcript packaged as a text
        InputDocument.

        Args:
            transcript: The raw transcript text.

        Returns:
            A SessionInputs instance ready for pipeline execution.
        """
        content = transcript.encode("utf-8")
        document = InputDocument(
            filename="evaluation_transcript.txt",
            document_type=DocumentType.TRANSCRIPT_TXT,
            content=content,
            size_bytes=len(content),
        )
        return SessionInputs(
            session_id=f"eval-{uuid.uuid4().hex[:8]}",
            documents=[document],
            backlog_tickets=[],
        )

    def _build_report(self, results: list[EvaluationCaseResult]) -> EvaluationReport:
        """Build the final evaluation report from per-case results.

        Computes aggregate metrics: mean and minimum keyword overlap, and
        mean judge scores (relevance, completeness, clarity) across all cases.

        Args:
            results: List of per-case evaluation results.

        Returns:
            A complete EvaluationReport with per-case and aggregate metrics.
        """
        keyword_scores = [r.keyword_overlap_score for r in results]

        # Compute aggregate keyword overlap metrics
        if keyword_scores:
            aggregate_keyword_mean = sum(keyword_scores) / len(keyword_scores)
            aggregate_keyword_min = min(keyword_scores)
        else:
            aggregate_keyword_mean = 0.0
            aggregate_keyword_min = 0.0

        # Compute aggregate judge score metrics
        judge_results = [r for r in results if r.judge_scores is not None]

        aggregate_relevance_mean: float | None = None
        aggregate_completeness_mean: float | None = None
        aggregate_clarity_mean: float | None = None

        if judge_results:
            aggregate_relevance_mean = sum(
                r.judge_scores.relevance for r in judge_results  # type: ignore[union-attr]
            ) / len(judge_results)
            aggregate_completeness_mean = sum(
                r.judge_scores.completeness for r in judge_results  # type: ignore[union-attr]
            ) / len(judge_results)
            aggregate_clarity_mean = sum(
                r.judge_scores.clarity for r in judge_results  # type: ignore[union-attr]
            ) / len(judge_results)

        return EvaluationReport(
            results=results,
            aggregate_keyword_overlap_mean=aggregate_keyword_mean,
            aggregate_keyword_overlap_min=aggregate_keyword_min,
            aggregate_relevance_mean=aggregate_relevance_mean,
            aggregate_completeness_mean=aggregate_completeness_mean,
            aggregate_clarity_mean=aggregate_clarity_mean,
        )
