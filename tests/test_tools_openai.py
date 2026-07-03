"""Unit tests for OpenAIGenerationTool with mocked openai SDK."""

from unittest.mock import MagicMock, patch

import pytest

# Build fake exception hierarchy that mirrors openai SDK
_OpenAIError = type("OpenAIError", (Exception,), {})
_APIError = type("APIError", (_OpenAIError,), {"message": "", "status_code": 0})
_APIStatusError = type("APIStatusError", (_APIError,), {})
_AuthenticationError = type("AuthenticationError", (_APIStatusError,), {})
_RateLimitError = type("RateLimitError", (_APIStatusError,), {})
_APIConnectionError = type("APIConnectionError", (_APIError,), {})
_APITimeoutError = type("APITimeoutError", (_APIConnectionError,), {})


def _make_mock_openai():
    """Build a mock openai module with real exception classes."""
    mock = MagicMock()
    mock.AuthenticationError = _AuthenticationError
    mock.RateLimitError = _RateLimitError
    mock.APIConnectionError = _APIConnectionError
    mock.APITimeoutError = _APITimeoutError
    mock.APIStatusError = _APIStatusError
    mock.OpenAI.return_value = MagicMock()
    return mock


@pytest.fixture
def setup():
    """Patch openai and return (tool, mock_openai)."""
    mock = _make_mock_openai()
    with patch("backlog_synthesizer.tools.llm_generation.openai", mock):
        from backlog_synthesizer.tools.llm_generation import OpenAIGenerationTool

        tool = OpenAIGenerationTool(api_key="test-key")
        yield tool, mock


class TestOpenAIGenerationTool:
    def test_successful_generation(self, setup):
        tool, _ = setup
        message = MagicMock()
        message.content = "Generated text"
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        tool._client.chat.completions.create.return_value = response

        result = tool.generate("Say hello", system_prompt="Be helpful")
        assert result == "Generated text"

    def test_authentication_error(self, setup):
        from backlog_synthesizer.tools.errors import PermanentToolError

        tool, _ = setup
        exc = _AuthenticationError("auth failed")
        exc.message = "auth failed"
        tool._client.chat.completions.create.side_effect = exc

        with pytest.raises(PermanentToolError, match="Authentication failed"):
            tool.generate("test")

    def test_rate_limit_error(self, setup):
        from backlog_synthesizer.tools.errors import TransientToolError

        tool, _ = setup
        exc = _RateLimitError("rate limited")
        exc.message = "rate limited"
        tool._client.chat.completions.create.side_effect = exc

        with pytest.raises(TransientToolError, match="Rate limit"):
            tool.generate("test")

    def test_api_connection_error(self, setup):
        from backlog_synthesizer.tools.errors import TransientToolError

        tool, _ = setup
        exc = _APIConnectionError("connection failed")
        exc.message = "connection failed"
        tool._client.chat.completions.create.side_effect = exc

        with pytest.raises(TransientToolError, match="Connection error"):
            tool.generate("test")

    def test_api_timeout_error(self, setup):
        from backlog_synthesizer.tools.errors import TransientToolError

        tool, _ = setup
        exc = _APITimeoutError("timed out")
        exc.message = "timed out"
        tool._client.chat.completions.create.side_effect = exc

        with pytest.raises(TransientToolError, match="timed out"):
            tool.generate("test")

    def test_api_status_error_500(self, setup):
        from backlog_synthesizer.tools.errors import TransientToolError

        tool, _ = setup
        exc = _APIStatusError("server error")
        exc.status_code = 500
        exc.message = "server error"
        tool._client.chat.completions.create.side_effect = exc

        with pytest.raises(TransientToolError, match="Transient API error"):
            tool.generate("test")

    def test_api_status_error_401(self, setup):
        from backlog_synthesizer.tools.errors import PermanentToolError

        tool, _ = setup
        exc = _APIStatusError("unauthorized")
        exc.status_code = 401
        exc.message = "unauthorized"
        tool._client.chat.completions.create.side_effect = exc

        with pytest.raises(PermanentToolError, match="Permanent API error"):
            tool.generate("test")

    def test_no_api_key(self):
        from backlog_synthesizer.tools.errors import PermanentToolError

        mock = _make_mock_openai()
        with patch("backlog_synthesizer.tools.llm_generation.openai", mock):
            with patch("backlog_synthesizer.tools.llm_generation.os.environ.get", return_value=None):
                from backlog_synthesizer.tools.llm_generation import OpenAIGenerationTool

                with pytest.raises(PermanentToolError, match="No API key"):
                    OpenAIGenerationTool(api_key=None)
