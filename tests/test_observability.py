"""Tests for observability module — tracing, metrics, and session replay."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from backlog_synthesizer.observability.tracing import PipelineTracer
from backlog_synthesizer.observability.metrics import PipelineMetrics, TOKEN_COST_RATES


class TestPipelineTracer:
    def test_record_agent_span(self):
        tracer = PipelineTracer("test-session", {"doc_count": 1})
        tracer.record_agent_span("agent.parser", {"items": 5}, 100)
        assert len(tracer._spans) == 1
        assert tracer._spans[0]["name"] == "agent.parser"
        assert tracer._spans[0]["duration_ms"] == 100

    def test_record_multiple_spans(self):
        tracer = PipelineTracer("test-session")
        tracer.record_agent_span("agent.parser", {"items": 5}, 100)
        tracer.record_agent_span("agent.gap_detection", {"dups": 2}, 200)
        tracer.record_agent_span("agent.story_writer", {"stories": 3}, 150)
        assert len(tracer._spans) == 3

    def test_record_span_with_error(self):
        tracer = PipelineTracer("test-session")
        tracer.record_agent_span("agent.parser", {}, 50, error="timeout")
        assert tracer._spans[0]["error"] == "timeout"

    def test_save_session_replay_creates_file(self, tmp_path):
        with patch("backlog_synthesizer.observability.tracing.RUNS_DIR", str(tmp_path)):
            with patch("backlog_synthesizer.observability.tracing.is_enabled", return_value=True):
                tracer = PipelineTracer("replay-test-123")
                tracer.record_agent_span("agent.parser", {"items": 3}, 50)
                tracer.save_session_replay("completed", {"total_tokens": 100})

                replay_file = tmp_path / "replay-test-123.json"
                assert replay_file.exists()
                data = json.loads(replay_file.read_text())
                assert data["session_id"] == "replay-test-123"
                assert data["status"] == "completed"
                assert len(data["spans"]) == 1
                assert data["metrics"]["total_tokens"] == 100

    def test_save_disabled_does_nothing(self, tmp_path):
        with patch("backlog_synthesizer.observability.tracing.RUNS_DIR", str(tmp_path)):
            with patch("backlog_synthesizer.observability.tracing.is_enabled", return_value=False):
                tracer = PipelineTracer("disabled-test")
                tracer.save_session_replay("completed")
                assert not (tmp_path / "disabled-test.json").exists()

    def test_session_id_and_timestamps(self):
        tracer = PipelineTracer("my-session-42", {"key": "value"})
        assert tracer.session_id == "my-session-42"
        assert tracer.started_at is not None
        assert tracer._root_attributes == {"key": "value"}


class TestPipelineMetrics:
    def test_record_tokens(self):
        m = PipelineMetrics()
        m.record_tokens("parser", 500)
        m.record_tokens("parser", 300)
        assert m._token_usage["parser"] == 800

    def test_record_tokens_multiple_agents(self):
        m = PipelineMetrics()
        m.record_tokens("parser", 500)
        m.record_tokens("story_writer", 1000)
        assert m._token_usage["parser"] == 500
        assert m._token_usage["story_writer"] == 1000

    def test_record_latency(self):
        m = PipelineMetrics()
        m.record_latency("parser", 150)
        assert m._latencies["parser"] == 150

    def test_record_items(self):
        m = PipelineMetrics()
        m.record_items("duplicate", 3)
        m.record_items("new", 7)
        assert m._item_counts["duplicate"] == 3
        assert m._item_counts["new"] == 7

    def test_record_error(self):
        m = PipelineMetrics()
        m.record_error("timeout")
        m.record_error("timeout")
        m.record_error("auth_failure")
        assert m._error_counts["timeout"] == 2
        assert m._error_counts["auth_failure"] == 1

    def test_record_reasoning_trigger(self):
        m = PipelineMetrics()
        m.record_reasoning_trigger("after_parser_empty")
        m.record_reasoning_trigger("after_parser_empty")
        assert m._reasoning_triggers["after_parser_empty"] == 2

    def test_estimated_cost_haiku(self):
        m = PipelineMetrics()
        m.set_model("claude-haiku-4-20250414")
        m.record_tokens("parser", 1000)
        # 1000 tokens * $0.00025/1K = $0.00025
        assert abs(m.estimated_cost_usd - 0.00025) < 0.0001

    def test_estimated_cost_unknown_model(self):
        m = PipelineMetrics()
        m.set_model("some-unknown-model")
        m.record_tokens("parser", 1000)
        # defaults to sonnet rate: 1000 * $0.003/1K = $0.003
        assert abs(m.estimated_cost_usd - 0.003) < 0.0001

    def test_estimated_cost_gpt4o_mini(self):
        m = PipelineMetrics()
        m.set_model("gpt-4o-mini")
        m.record_tokens("parser", 2000)
        # 2000 tokens * $0.00015/1K = $0.0003
        assert abs(m.estimated_cost_usd - 0.0003) < 0.0001

    def test_to_dict(self):
        m = PipelineMetrics()
        m.set_model("gpt-4o-mini")
        m.record_tokens("parser", 100)
        m.record_latency("parser", 50)
        m.record_items("new", 5)
        d = m.to_dict()
        assert d["total_tokens"] == 100
        assert d["model"] == "gpt-4o-mini"
        assert "estimated_cost_usd" in d
        assert d["latencies_ms"]["parser"] == 50
        assert d["item_counts"]["new"] == 5

    def test_to_dict_empty(self):
        m = PipelineMetrics()
        d = m.to_dict()
        assert d["total_tokens"] == 0
        assert d["estimated_cost_usd"] == 0.0
        assert d["model"] == ""


class TestOtelDisabled:
    def test_disabled_env_var(self):
        with patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
            # Re-import to pick up env change
            import importlib
            import backlog_synthesizer.observability as obs
            importlib.reload(obs)
            assert not obs._OTEL_ENABLED
            # Restore
            with patch.dict(os.environ, {"OTEL_ENABLED": "true"}):
                importlib.reload(obs)

    def test_tracer_works_without_otel_enabled(self):
        """PipelineTracer and PipelineMetrics work even without OTel being enabled."""
        tracer = PipelineTracer("standalone-test")
        tracer.record_agent_span("agent.parser", {"items": 1}, 10)
        assert len(tracer._spans) == 1

        metrics = PipelineMetrics()
        metrics.record_tokens("parser", 100)
        metrics.record_latency("parser", 50)
        d = metrics.to_dict()
        assert d["total_tokens"] == 100
