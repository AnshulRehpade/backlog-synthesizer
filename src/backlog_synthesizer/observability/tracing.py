"""Pipeline tracing — one trace per pipeline run with child spans per agent."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backlog_synthesizer.observability import is_enabled

RUNS_DIR = os.environ.get("RUNS_DIR", "runs")


class PipelineTracer:
    """Records spans and attributes for a single pipeline run."""

    def __init__(self, session_id: str, attributes: dict[str, Any] | None = None):
        self.session_id = session_id
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._spans: list[dict] = []
        self._root_attributes = attributes or {}

    def record_agent_span(
        self,
        agent_name: str,
        attributes: dict[str, Any],
        duration_ms: int,
        error: str | None = None,
    ) -> None:
        """Record a completed agent span."""
        span_data = {
            "name": agent_name,
            "attributes": attributes,
            "duration_ms": duration_ms,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._spans.append(span_data)

    def save_session_replay(self, status: str, metrics: dict | None = None) -> None:
        """Serialize the trace to a JSON file for session replay."""
        if not is_enabled():
            return

        runs_dir = Path(RUNS_DIR)
        runs_dir.mkdir(parents=True, exist_ok=True)

        replay = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "root_attributes": self._root_attributes,
            "spans": self._spans,
            "metrics": metrics or {},
        }

        replay_path = runs_dir / f"{self.session_id}.json"
        replay_path.write_text(json.dumps(replay, indent=2, default=str))
