"""Evaluation CLI — run, filter, and compare evaluation results."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from backlog_synthesizer.evaluation.regression import (
    detect_regression,
    load_latest_baseline,
    save_results,
)
from backlog_synthesizer.models.evaluation import (
    EvalRunResult,
    EvalSummary,
    GoldenDatasetEntry,
)

GOLDEN_DATASET_PATH = Path("data/golden_dataset/golden_entries.json")


def main() -> None:
    """Entry point for the evaluation CLI."""
    parser = argparse.ArgumentParser(description="Backlog Synthesizer Evaluation")
    subparsers = parser.add_subparsers(dest="command")

    # run command
    run_parser = subparsers.add_parser("run", help="Run evaluation against golden dataset")
    run_parser.add_argument("--tag", help="Filter golden entries by tag")
    run_parser.add_argument("--id", help="Run single golden entry by ID")
    run_parser.add_argument("--threshold-keyword", type=float, default=0.60)
    run_parser.add_argument("--threshold-f1", type=float, default=0.60)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate golden entries load correctly without running pipeline",
    )
    run_parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current results as the regression baseline",
    )

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two session replays")
    compare_parser.add_argument("file_a", help="First session replay JSON")
    compare_parser.add_argument("file_b", help="Second session replay JSON")

    args = parser.parse_args()

    if args.command == "run":
        run_evaluation(args)
    elif args.command == "compare":
        compare_runs(args)
    else:
        parser.print_help()
        sys.exit(1)


def load_golden_entries(
    tag: str | None = None, entry_id: str | None = None
) -> list[GoldenDatasetEntry]:
    """Load and optionally filter golden dataset entries.

    Args:
        tag: If provided, only entries containing this tag are returned.
        entry_id: If provided, only the entry with this ID is returned.

    Returns:
        List of validated GoldenDatasetEntry objects.

    Raises:
        SystemExit: If the golden dataset file is missing or invalid.
    """
    if not GOLDEN_DATASET_PATH.exists():
        print(f"ERROR: Golden dataset not found at {GOLDEN_DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        raw = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in golden dataset: {e}", file=sys.stderr)
        sys.exit(1)

    entries = [GoldenDatasetEntry.model_validate(item) for item in raw]

    if entry_id:
        entries = [e for e in entries if e.id == entry_id]
    if tag:
        entries = [e for e in entries if tag in e.tags]

    return entries


def compute_keyword_overlap(generated_tokens: list[str], expected_tokens: list[str]) -> float:
    """Compute normalized keyword overlap between generated and expected token lists.

    Args:
        generated_tokens: Tokens from generated output.
        expected_tokens: Tokens from expected output.

    Returns:
        Float between 0.0 and 1.0. Returns 1.0 if expected is empty.
    """
    if not expected_tokens:
        return 1.0

    generated_lower = {t.lower() for t in generated_tokens}
    expected_lower = [t.lower() for t in expected_tokens]

    matches = sum(1 for t in expected_lower if t in generated_lower)
    return matches / len(expected_lower)


async def _evaluate_entry(
    orchestrator, entry: GoldenDatasetEntry, threshold: float
) -> EvalRunResult:
    """Run a single golden entry through the real pipeline and score it.

    Args:
        orchestrator: The OrchestratorAgent instance.
        entry: The golden dataset entry to evaluate.
        threshold: Keyword overlap threshold for pass/fail.

    Returns:
        An EvalRunResult with computed metrics.
    """
    from backlog_synthesizer.models.inputs import (
        BacklogTicket,
        DocumentType,
        InputDocument,
        SessionInputs,
    )
    import uuid

    # Build session inputs from the golden entry
    content = entry.transcript.encode("utf-8") if entry.transcript else b""
    documents = []
    if content:
        documents.append(InputDocument(
            filename="eval_transcript.txt",
            document_type=DocumentType.TRANSCRIPT_TXT,
            content=content,
            size_bytes=len(content),
        ))

    # Parse existing_backlog into BacklogTicket objects
    backlog_tickets = []
    for ticket_data in entry.existing_backlog:
        try:
            ticket = BacklogTicket.model_validate(ticket_data)
            backlog_tickets.append(ticket)
        except Exception:
            pass  # Skip invalid tickets

    inputs = SessionInputs(
        session_id=f"eval-{uuid.uuid4().hex[:8]}",
        documents=documents,
        backlog_tickets=backlog_tickets,
    )

    # Run pipeline
    result = await orchestrator.run_session(inputs)

    # Count generated stories
    stories_generated = 0
    if result.output:
        for epic in result.output.epics:
            stories_generated += len(epic.stories)

    # Compute keyword overlap between generated story text and transcript
    generated_tokens = []
    if result.output:
        for epic in result.output.epics:
            for story in epic.stories:
                generated_tokens.extend(story.user_story.split())
                for ac in story.acceptance_criteria:
                    generated_tokens.extend(ac.description.split())

    # Expected tokens from the transcript (what should be captured)
    expected_tokens = entry.transcript.split() if entry.transcript else []

    keyword_score = compute_keyword_overlap(generated_tokens, expected_tokens)

    # Determine pass/fail
    passed = True
    failure_reason = None

    if keyword_score < threshold:
        passed = False
        failure_reason = f"keyword_overlap {keyword_score:.3f} below threshold {threshold}"

    # Check story count expectations (if provided)
    if entry.expected_stories_count > 0 and stories_generated == 0 and entry.transcript:
        passed = False
        failure_reason = f"Expected {entry.expected_stories_count} stories, got 0"

    return EvalRunResult(
        golden_id=entry.id,
        keyword_overlap=keyword_score,
        stories_generated=stories_generated,
        passed=passed,
        failure_reason=failure_reason,
    )


def run_evaluation(args: argparse.Namespace) -> None:
    """Execute evaluation against the golden dataset.

    In dry-run mode, validates that entries load correctly.
    In normal mode, instantiates the real pipeline, runs each golden entry
    through OrchestratorAgent.run_session(), and scores using keyword overlap.

    Args:
        args: Parsed CLI arguments.
    """
    entries = load_golden_entries(tag=args.tag, entry_id=args.id)

    if not entries:
        print("No golden entries matched the filter criteria.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"Dry run: successfully loaded {len(entries)} golden entries.")
        for entry in entries:
            print(f"  {entry.id}: {entry.description} (tags: {', '.join(entry.tags)})")
        sys.exit(0)

    # Initialize real pipeline
    try:
        from backlog_synthesizer.main import create_pipeline
        orchestrator = create_pipeline()
    except Exception as e:
        print(f"Pipeline initialization failed: {e}", file=sys.stderr)
        print("Use --dry-run for validation without API calls.", file=sys.stderr)
        sys.exit(1)

    # Run evaluation through real pipeline
    results: list[EvalRunResult] = []
    any_failed = False
    total_latency_ms = 0

    for entry in entries:
        print(f"  Evaluating {entry.id}...", end=" ", flush=True)
        start_time = time.time()

        try:
            result = asyncio.run(_evaluate_entry(orchestrator, entry, args.threshold_keyword))
            duration_ms = int((time.time() - start_time) * 1000)
            total_latency_ms += duration_ms

            if not result.passed:
                any_failed = True
            results.append(result)

            status = "✅" if result.passed else "❌"
            print(f"{status} ({result.keyword_overlap:.3f}, {duration_ms}ms)")
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            total_latency_ms += duration_ms
            print(f"❌ ERROR: {e}")
            results.append(EvalRunResult(
                golden_id=entry.id,
                keyword_overlap=0.0,
                stories_generated=0,
                passed=False,
                failure_reason=str(e),
            ))
            any_failed = True

    # Compute aggregate stats
    scores = [r.keyword_overlap for r in results]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    std_score = _compute_std(scores, mean_score)
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    success_rate = sum(1 for r in results if r.passed) / len(results) if results else 0.0

    summary = EvalSummary(
        total_entries=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        keyword_overlap_mean=mean_score,
        keyword_overlap_std=std_score,
        keyword_overlap_min=min_score,
        keyword_overlap_max=max_score,
        success_rate=success_rate,
        results=results,
    )

    # Print results table
    _print_results_table(results)
    _print_summary(summary)

    # Check for regressions against baseline
    current_metrics = {
        "keyword_overlap_mean": mean_score,
        "success_rate": success_rate,
    }
    baseline = load_latest_baseline()
    regression = detect_regression(current_metrics, baseline)

    if regression.has_regression:
        print("\n⚠️  REGRESSION DETECTED:")
        for reg in regression.regressions:
            print(
                f"  {reg['metric']}: {reg['baseline']:.3f} → {reg['current']:.3f} "
                f"(Δ {reg['diff']:+.3f}, threshold: {reg['threshold']})"
            )

    if regression.improvements:
        print("\n✅ IMPROVEMENTS:")
        for imp in regression.improvements:
            print(
                f"  {imp['metric']}: {imp['baseline']:.3f} → {imp['current']:.3f} "
                f"(Δ {imp['diff']:+.3f})"
            )

    # Save results to history
    save_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "keyword_overlap_mean": mean_score,
        "keyword_overlap_std": std_score,
        "keyword_overlap_min": min_score,
        "keyword_overlap_max": max_score,
        "success_rate": success_rate,
        "total_entries": len(results),
        "passed": summary.passed,
        "failed": summary.failed,
    }
    saved_path = save_results(save_data)
    print(f"\nResults saved to: {saved_path}")

    # Save baseline if requested
    if hasattr(args, 'save_baseline') and args.save_baseline:
        baseline_path = Path("evaluation/history/baseline_results.json")
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(save_data, indent=2))
        print(f"Baseline saved to: {baseline_path}")

    if any_failed:
        sys.exit(1)


def compare_runs(args: argparse.Namespace) -> None:
    """Compare two evaluation result files side by side.

    Args:
        args: Parsed CLI arguments with file_a and file_b paths.
    """
    path_a = Path(args.file_a)
    path_b = Path(args.file_b)

    if not path_a.exists():
        print(f"ERROR: File not found: {path_a}", file=sys.stderr)
        sys.exit(1)
    if not path_b.exists():
        print(f"ERROR: File not found: {path_b}", file=sys.stderr)
        sys.exit(1)

    try:
        data_a = json.loads(path_a.read_text(encoding="utf-8"))
        data_b = json.loads(path_b.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("EVALUATION COMPARISON")
    print("=" * 60)
    print(f"  File A: {path_a.name}")
    print(f"  File B: {path_b.name}")
    print("-" * 60)
    print(f"{'Metric':<30} {'File A':>10} {'File B':>10} {'Diff':>10}")
    print("-" * 60)

    all_keys = sorted(set(list(data_a.keys()) + list(data_b.keys())))
    for key in all_keys:
        val_a = data_a.get(key)
        val_b = data_b.get(key)

        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            diff = val_b - val_a
            print(f"  {key:<28} {val_a:>10.4f} {val_b:>10.4f} {diff:>+10.4f}")
        elif val_a is not None and val_b is not None:
            print(f"  {key:<28} {str(val_a):>10} {str(val_b):>10} {'—':>10}")
        else:
            a_str = str(val_a) if val_a is not None else "N/A"
            b_str = str(val_b) if val_b is not None else "N/A"
            print(f"  {key:<28} {a_str:>10} {b_str:>10} {'—':>10}")

    print("=" * 60)


def _print_results_table(results: list[EvalRunResult]) -> None:
    """Print a formatted table of per-entry results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'ID':<14} {'Overlap':>8} {'Stories':>8} {'Status':>8} {'Reason'}")
    print("-" * 70)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        reason = r.failure_reason or ""
        print(f"  {r.golden_id:<12} {r.keyword_overlap:>8.3f} {r.stories_generated:>8} {status:>8}   {reason}")

    print("-" * 70)


def _print_summary(summary: EvalSummary) -> None:
    """Print aggregate evaluation statistics."""
    print("\nAGGREGATE STATS:")
    print(f"  Total entries: {summary.total_entries}")
    print(f"  Passed:        {summary.passed}")
    print(f"  Failed:        {summary.failed}")
    print(f"  Success rate:  {summary.success_rate:.2%}")
    print(f"  Keyword overlap — mean: {summary.keyword_overlap_mean:.3f}, "
          f"std: {summary.keyword_overlap_std:.3f}, "
          f"min: {summary.keyword_overlap_min:.3f}, "
          f"max: {summary.keyword_overlap_max:.3f}")


def _compute_std(values: list[float], mean: float) -> float:
    """Compute standard deviation of a list of values.

    Args:
        values: List of numeric values.
        mean: Pre-computed mean of the values.

    Returns:
        Standard deviation, or 0.0 if fewer than 2 values.
    """
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
