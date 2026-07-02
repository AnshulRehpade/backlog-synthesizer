"""Shared test fixtures and configuration for the Backlog Synthesizer test suite."""

import pytest


@pytest.fixture
def sample_session_id() -> str:
    """Provide a consistent test session identifier."""
    return "test-session-001"
