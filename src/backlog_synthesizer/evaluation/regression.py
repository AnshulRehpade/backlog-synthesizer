"""Regression detection — compares current eval results against historical baseline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HISTORY_DIR = Path("evaluation/history")

# Default thresholds — a metric must drop by more than this amount to be a regression
DEFAULT_THRESHOLDS = {
    "keyword_overlap_mean": 0.05,
    "f1_score": 0.05,
    "success_rate": 0.05,
}


class RegressionResult:
    """Result of comparing current evaluation results against a baseline."""

    def __init__(self) -> None:
        self.regressions: list[dict] = []
        self.improvements: list[dict] = []
        self.has_regression: bool = False


def detect_regression(
    current_results: dict,
    baseline_results: dict | None = None,
    thresholds: dict | None = None,
) -> RegressionResult:
    """Compare current results against baseline and detect regressions.

    A regression is detected when a metric drops by more than the configured
    threshold compared to the baseline. If no baseline is provided, no
    regression can be detected.

    Args:
        current_results: Dictionary of current evaluation metrics.
        baseline_results: Dictionary of baseline evaluation metrics, or None.
        thresholds: Custom thresholds per metric. Defaults to DEFAULT_THRESHOLDS.

    Returns:
        A RegressionResult indicating any regressions or improvements found.
    """
    result = RegressionResult()

    if baseline_results is None:
        return result

    effective_thresholds = thresholds or DEFAULT_THRESHOLDS

    # Compare each metric that exists in both current and baseline
    comparable_metrics = set(current_results.keys()) & set(baseline_results.keys())

    for metric in comparable_metrics:
        current_value = current_results.get(metric)
        baseline_value = baseline_results.get(metric)

        # Skip non-numeric values
        if not isinstance(current_value, (int, float)) or not isinstance(
            baseline_value, (int, float)
        ):
            continue

        threshold = effective_thresholds.get(metric, 0.05)
        diff = current_value - baseline_value

        if diff < -threshold:
            result.regressions.append(
                {
                    "metric": metric,
                    "current": current_value,
                    "baseline": baseline_value,
                    "diff": diff,
                    "threshold": threshold,
                }
            )
        elif diff > threshold:
            result.improvements.append(
                {
                    "metric": metric,
                    "current": current_value,
                    "baseline": baseline_value,
                    "diff": diff,
                }
            )

    result.has_regression = len(result.regressions) > 0
    return result


def save_results(results: dict) -> Path:
    """Save evaluation results to history directory.

    Creates the history directory if it does not exist. The file is named
    with a timestamp for chronological ordering.

    Args:
        results: Dictionary of evaluation results to persist.

    Returns:
        Path to the saved results file.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_results.json"
    filepath = HISTORY_DIR / filename

    filepath.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return filepath


def load_latest_baseline() -> dict | None:
    """Load the most recent results file from history.

    Scans the history directory for JSON files, sorts by filename
    (which embeds the timestamp), and returns the contents of the
    most recent one.

    Returns:
        Dictionary of the most recent evaluation results, or None if
        no history exists.
    """
    if not HISTORY_DIR.exists():
        return None

    json_files = sorted(HISTORY_DIR.glob("*_results.json"))
    if not json_files:
        return None

    latest = json_files[-1]
    content = latest.read_text(encoding="utf-8")
    return json.loads(content)
