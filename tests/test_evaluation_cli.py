"""Tests for evaluation CLI, regression detection, and golden dataset loading."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backlog_synthesizer.evaluation.regression import (
    RegressionResult,
    detect_regression,
    load_latest_baseline,
    save_results,
)
from backlog_synthesizer.models.evaluation import (
    EvalRunResult,
    EvalSummary,
    GoldenDatasetEntry,
)


class TestGoldenDatasetLoading:
    """Tests for golden dataset file structure and validation."""

    def test_golden_entries_file_exists(self):
        path = Path("data/golden_dataset/golden_entries.json")
        assert path.exists()

    def test_golden_entries_valid_json(self):
        path = Path("data/golden_dataset/golden_entries.json")
        entries = json.loads(path.read_text())
        assert isinstance(entries, list)
        assert len(entries) >= 20

    def test_each_entry_has_required_fields(self):
        path = Path("data/golden_dataset/golden_entries.json")
        entries = json.loads(path.read_text())
        for entry in entries:
            assert "id" in entry
            assert "description" in entry
            assert "transcript" in entry
            assert "tags" in entry

    def test_entries_validate_as_pydantic(self):
        path = Path("data/golden_dataset/golden_entries.json")
        entries = json.loads(path.read_text())
        for entry in entries:
            validated = GoldenDatasetEntry.model_validate(entry)
            assert validated.id
            assert isinstance(validated.transcript, str)

    def test_entries_have_unique_ids(self):
        path = Path("data/golden_dataset/golden_entries.json")
        entries = json.loads(path.read_text())
        ids = [e["id"] for e in entries]
        assert len(ids) == len(set(ids)), "Golden entry IDs must be unique"

    def test_entries_cover_expected_tags(self):
        path = Path("data/golden_dataset/golden_entries.json")
        entries = json.loads(path.read_text())
        all_tags = set()
        for entry in entries:
            all_tags.update(entry["tags"])
        # Verify key tag categories exist
        assert "normal" in all_tags
        assert "edge_case" in all_tags
        assert "duplicates" in all_tags
        assert "conflicts" in all_tags
        assert "technical" in all_tags
        assert "minimal" in all_tags
        assert "vague" in all_tags

    def test_edge_case_entries_have_zero_expected_stories(self):
        path = Path("data/golden_dataset/golden_entries.json")
        entries = json.loads(path.read_text())
        edge_cases = [e for e in entries if "edge_case" in e["tags"] and "empty" in e["tags"]]
        for entry in edge_cases:
            assert entry["expected_stories_count"] == 0


class TestRegressionDetection:
    """Tests for the regression detection logic."""

    def test_no_regression_when_metrics_improve(self):
        current = {"keyword_overlap_mean": 0.80, "success_rate": 0.95}
        baseline = {"keyword_overlap_mean": 0.75, "success_rate": 0.90}
        result = detect_regression(current, baseline)
        assert not result.has_regression

    def test_regression_when_keyword_drops(self):
        current = {"keyword_overlap_mean": 0.55, "success_rate": 0.90}
        baseline = {"keyword_overlap_mean": 0.75, "success_rate": 0.90}
        result = detect_regression(current, baseline)
        assert result.has_regression
        assert len(result.regressions) >= 1

    def test_no_regression_without_baseline(self):
        current = {"keyword_overlap_mean": 0.50, "success_rate": 0.80}
        result = detect_regression(current, None)
        assert not result.has_regression

    def test_no_regression_within_threshold(self):
        current = {"keyword_overlap_mean": 0.72, "success_rate": 0.88}
        baseline = {"keyword_overlap_mean": 0.75, "success_rate": 0.90}
        result = detect_regression(current, baseline)
        assert not result.has_regression

    def test_improvement_detected(self):
        current = {"keyword_overlap_mean": 0.90, "success_rate": 0.98}
        baseline = {"keyword_overlap_mean": 0.75, "success_rate": 0.85}
        result = detect_regression(current, baseline)
        assert not result.has_regression
        assert len(result.improvements) >= 1

    def test_custom_thresholds(self):
        current = {"keyword_overlap_mean": 0.70, "success_rate": 0.85}
        baseline = {"keyword_overlap_mean": 0.75, "success_rate": 0.90}
        # With tight threshold of 0.01, even small drops are regressions
        result = detect_regression(
            current, baseline, thresholds={"keyword_overlap_mean": 0.01, "success_rate": 0.01}
        )
        assert result.has_regression
        assert len(result.regressions) == 2

    def test_save_and_load_results(self, tmp_path):
        with patch(
            "backlog_synthesizer.evaluation.regression.HISTORY_DIR", tmp_path
        ):
            results = {
                "keyword_overlap_mean": 0.75,
                "success_rate": 0.90,
                "timestamp": "2024-01-01",
            }
            path = save_results(results)
            assert path.exists()

            loaded = load_latest_baseline()
            assert loaded is not None
            assert loaded["keyword_overlap_mean"] == 0.75

    def test_load_latest_baseline_empty_dir(self, tmp_path):
        with patch(
            "backlog_synthesizer.evaluation.regression.HISTORY_DIR", tmp_path
        ):
            loaded = load_latest_baseline()
            assert loaded is None

    def test_load_latest_baseline_nonexistent_dir(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist"
        with patch(
            "backlog_synthesizer.evaluation.regression.HISTORY_DIR", nonexistent
        ):
            loaded = load_latest_baseline()
            assert loaded is None

    def test_save_creates_directory(self, tmp_path):
        new_dir = tmp_path / "new_history"
        with patch(
            "backlog_synthesizer.evaluation.regression.HISTORY_DIR", new_dir
        ):
            results = {"keyword_overlap_mean": 0.80}
            path = save_results(results)
            assert path.exists()
            assert new_dir.exists()


class TestEvalSummary:
    """Tests for the EvalSummary model."""

    def test_summary_from_results(self):
        summary = EvalSummary(
            total_entries=3,
            passed=2,
            failed=1,
            keyword_overlap_mean=0.7,
            keyword_overlap_std=0.1,
            keyword_overlap_min=0.5,
            keyword_overlap_max=0.9,
            success_rate=0.67,
        )
        assert summary.success_rate == 0.67
        assert summary.total_entries == 3

    def test_summary_with_results_list(self):
        results = [
            EvalRunResult(golden_id="golden_001", keyword_overlap=0.8, passed=True),
            EvalRunResult(
                golden_id="golden_002",
                keyword_overlap=0.4,
                passed=False,
                failure_reason="below threshold",
            ),
        ]
        summary = EvalSummary(
            total_entries=2,
            passed=1,
            failed=1,
            keyword_overlap_mean=0.6,
            keyword_overlap_std=0.2,
            keyword_overlap_min=0.4,
            keyword_overlap_max=0.8,
            success_rate=0.5,
            results=results,
        )
        assert len(summary.results) == 2
        assert summary.results[0].passed is True
        assert summary.results[1].passed is False

    def test_eval_run_result_defaults(self):
        result = EvalRunResult(golden_id="test_001")
        assert result.keyword_overlap == 0.0
        assert result.stories_generated == 0
        assert result.passed is True
        assert result.failure_reason is None


class TestGoldenDatasetEntry:
    """Tests for the GoldenDatasetEntry model."""

    def test_minimal_entry(self):
        entry = GoldenDatasetEntry(
            id="test_001",
            description="Test entry",
            transcript="Some transcript text",
        )
        assert entry.id == "test_001"
        assert entry.existing_backlog == []
        assert entry.expected_stories_count == 0
        assert entry.tags == []

    def test_full_entry(self):
        entry = GoldenDatasetEntry(
            id="test_002",
            description="Full entry with all fields",
            transcript="Meeting notes",
            existing_backlog=[
                {"id": "T-1", "title": "Existing ticket", "status": "open"}
            ],
            expected_stories_count=5,
            expected_duplicates=1,
            expected_conflicts=2,
            tags=["normal", "full"],
        )
        assert entry.expected_stories_count == 5
        assert entry.expected_duplicates == 1
        assert entry.expected_conflicts == 2
        assert len(entry.existing_backlog) == 1


class TestCLIHelpers:
    """Tests for CLI utility functions."""

    def test_compute_keyword_overlap_full_match(self):
        from backlog_synthesizer.evaluation.cli import compute_keyword_overlap

        tokens = ["hello", "world", "test"]
        score = compute_keyword_overlap(tokens, tokens)
        assert score == 1.0

    def test_compute_keyword_overlap_no_match(self):
        from backlog_synthesizer.evaluation.cli import compute_keyword_overlap

        generated = ["alpha", "beta", "gamma"]
        expected = ["delta", "epsilon", "zeta"]
        score = compute_keyword_overlap(generated, expected)
        assert score == 0.0

    def test_compute_keyword_overlap_partial(self):
        from backlog_synthesizer.evaluation.cli import compute_keyword_overlap

        generated = ["hello", "world", "extra"]
        expected = ["hello", "world", "missing"]
        score = compute_keyword_overlap(generated, expected)
        assert abs(score - 2.0 / 3.0) < 0.001

    def test_compute_keyword_overlap_empty_expected(self):
        from backlog_synthesizer.evaluation.cli import compute_keyword_overlap

        score = compute_keyword_overlap(["some", "tokens"], [])
        assert score == 1.0

    def test_compute_keyword_overlap_case_insensitive(self):
        from backlog_synthesizer.evaluation.cli import compute_keyword_overlap

        generated = ["Hello", "WORLD"]
        expected = ["hello", "world"]
        score = compute_keyword_overlap(generated, expected)
        assert score == 1.0

    def test_load_golden_entries_all(self):
        from backlog_synthesizer.evaluation.cli import load_golden_entries

        entries = load_golden_entries()
        assert len(entries) >= 20

    def test_load_golden_entries_filter_by_tag(self):
        from backlog_synthesizer.evaluation.cli import load_golden_entries

        entries = load_golden_entries(tag="edge_case")
        assert len(entries) >= 2
        for entry in entries:
            assert "edge_case" in entry.tags

    def test_load_golden_entries_filter_by_id(self):
        from backlog_synthesizer.evaluation.cli import load_golden_entries

        entries = load_golden_entries(entry_id="golden_001")
        assert len(entries) == 1
        assert entries[0].id == "golden_001"
