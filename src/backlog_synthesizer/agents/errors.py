"""Custom exception classes for the Orchestrator Agent pipeline.

These exceptions communicate pipeline-level failure modes:
- PipelineHaltError: raised when a permanent error halts the pipeline immediately
- RetryExhaustedError: raised when all retries are exhausted for a transient error

Requirements: 2.5, 2.6, 7.1, 7.2, 7.3
"""


class PipelineHaltError(Exception):
    """Raised when a permanent error halts the pipeline immediately.

    Permanent errors include HTTP 401, 403, 404, authentication failures,
    and schema violations. These are not retried.

    Attributes:
        original_error: The underlying error that triggered the halt.
    """

    def __init__(self, original_error: Exception) -> None:
        super().__init__(f"Pipeline halted due to permanent error: {original_error}")
        self.original_error = original_error


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted for a transient error.

    Transient errors include HTTP 429, 500, 502, 503, 504, network timeouts,
    and sub-agent timeouts. After max_retries attempts, this exception is raised.

    Attributes:
        original_error: The last transient error encountered.
        attempts: Total number of attempts made (including the initial attempt).
    """

    def __init__(self, original_error: Exception, attempts: int = 4) -> None:
        super().__init__(
            f"Retries exhausted after {attempts} attempts: {original_error}"
        )
        self.original_error = original_error
        self.attempts = attempts
