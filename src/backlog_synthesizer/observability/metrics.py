"""Pipeline metrics — counters and histograms for key pipeline indicators."""

# Token cost rates per 1K input tokens
TOKEN_COST_RATES = {
    "claude-haiku-4-20250414": 0.00025,
    "claude-haiku-4-5": 0.00025,
    "claude-sonnet-4-20250514": 0.003,
    "gpt-4o-mini": 0.00015,
}
DEFAULT_RATE = 0.003  # default to sonnet rate


class PipelineMetrics:
    """Collects metrics for a single pipeline run."""

    def __init__(self):
        self._token_usage: dict[str, int] = {}
        self._latencies: dict[str, int] = {}
        self._item_counts: dict[str, int] = {}
        self._error_counts: dict[str, int] = {}
        self._reasoning_triggers: dict[str, int] = {}
        self._model: str = ""

    def set_model(self, model: str) -> None:
        self._model = model

    def record_tokens(self, agent: str, count: int) -> None:
        self._token_usage[agent] = self._token_usage.get(agent, 0) + count

    def record_latency(self, agent: str, duration_ms: int) -> None:
        self._latencies[agent] = duration_ms

    def record_items(self, classification: str, count: int) -> None:
        self._item_counts[classification] = count

    def record_error(self, error_type: str) -> None:
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

    def record_reasoning_trigger(self, decision_point: str) -> None:
        self._reasoning_triggers[decision_point] = self._reasoning_triggers.get(decision_point, 0) + 1

    @property
    def estimated_cost_usd(self) -> float:
        total_tokens = sum(self._token_usage.values())
        rate = TOKEN_COST_RATES.get(self._model, DEFAULT_RATE)
        return (total_tokens / 1000) * rate

    def to_dict(self) -> dict:
        return {
            "token_usage": self._token_usage,
            "latencies_ms": self._latencies,
            "item_counts": self._item_counts,
            "error_counts": self._error_counts,
            "reasoning_triggers": self._reasoning_triggers,
            "total_tokens": sum(self._token_usage.values()),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "model": self._model,
        }
