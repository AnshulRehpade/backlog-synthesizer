"""Unit tests for AnthropicGenerationTool with mocked anthropic SDK."""

from unittest.mock import MagicMock, patch

import pytest

# Build fake exception hierarchy that mirrors anthropic SDK
_AnthropicError = type("AnthropicError", (Exception,), {})
_APIError = type("APIError", (_AnthropicError,), {"message": "", "status_code": 0, "body": None})
_APIStatusError = type("APIStatusError", (_APIError,), {})
_AuthenticationError = type("AuthenticationError", (_APIStatusError,), {})
_RateLimitError = type("RateLimitError", (_APIStatusError,), {})
_APIConnectionError = type("APIConnectionError", (_APIError,), {})
_APITimeoutError = type("APITimeoutError", (_APIConnectionError,), {})


def _make_mock_anthropic():
    """Build a mock anthropic module with real exception classes."""
    mock = MagicMock()
    mock.AuthenticationError = _AuthenticationError
    mock.RateLimitError = _RateLimitError
    mock.APIConnectionError = _APIConnectionError
    mock.APITimeoutError = _APITimeoutError
    mock.APIStatusError = _APIStatusError
    mock.Anthropic.return_value = MagicMock()
    return mock


@pytest.fixture
def setup():
    """Patch anthropic and return (tool, mock_anthropic)."""
    mock = _make_mock_anthropic()
    with patch("backlog_synthesizer.tools.anthropic_generation.anthropic", mock):
        from backlog_synthesizer.tools.anthropic_generation import AnthropicGenerationTool

        tool = AnthropicGenerationTool(api_key="test-key")
        yield tool, mock


class TestAnthropicGenerationTool:
    def test_successful_generation(self, setup):
        tool, mock = setup
        block = MagicMock()
        block.type = "text"
        block.text = "Hello world"
        response = MagicMock()
        response.content = [block]
        tool._client.messages.create.return_value = response

        result = tool.generate("Say hello")
        assert result == "Hello world"

    def test_authentication_error(self, setup):
        from backlog_synthesizer.tools.errors import PermanentToolError

        tool, _ = setup
        exc = _AuthenticationError("auth failed")
        exc.message = "auth failed"
        tool._client.messages.create.side_effect = exc

        with pytest.raises(PermanentToolError, match="Authentication failed"):
            tool.generate("test")

    def test_rate_limit_error(self, setup):
        from backlog_synthesizer.tools.errors import TransientToolError

        tool, _ = setup
        exc = _RateLimitError("rate limited")
        exc.message = "rate limited"
        tool._client.messages.create.side_effect = exc

        with pytest.raises(TransientToolError, match="Rate limit"):
            tool.generate("test")

    def test_api_connection_error(self, setup):
        from backlog_synthesizer.tools.errors import TransientToolError

        tool, _ = setup
        exc = _APIConnectionError("connection failed")
        tool._client.messages.create.side_effect = exc

        with pytest.raises(TransientToolError, match="Connection error"):
            tool.generate("test")

    def test_api_timeout_error(self, setup):
        from backlog_synthesizer.tools.errors import TransientToolError

        tool, _ = setup
        exc = _APITimeoutError("timed out")
        tool._client.messages.create.side_effect = exc

        with pytest.raises(TransientToolError, match="timed out"):
            tool.generate("test")

    def test_api_status_error_500(self, setup):
        from backlog_synthesizer.tools.errors import TransientToolError

        tool, _ = setup
        exc = _APIStatusError("server error")
        exc.status_code = 500
        exc.message = "server error"
        tool._client.messages.create.side_effect = exc

        with pytest.raises(TransientToolError, match="Transient API error"):
            tool.generate("test")

    def test_api_status_error_401(self, setup):
        from backlog_synthesizer.tools.errors import PermanentToolError

        tool, _ = setup
        exc = _APIStatusError("unauthorized")
        exc.status_code = 401
        exc.message = "unauthorized"
        tool._client.messages.create.side_effect = exc

        with pytest.raises(PermanentToolError, match="Permanent API error"):
            tool.generate("test")

    def test_empty_content(self, setup):
        from backlog_synthesizer.tools.errors import ToolError

        tool, _ = setup
        response = MagicMock()
        response.content = []
        tool._client.messages.create.return_value = response

        with pytest.raises(ToolError, match="empty content"):
            tool.generate("test")

    def test_no_api_key(self):
        from backlog_synthesizer.tools.errors import PermanentToolError

        mock = _make_mock_anthropic()
        with patch("backlog_synthesizer.tools.anthropic_generation.anthropic", mock):
            with patch("backlog_synthesizer.tools.anthropic_generation.os.environ.get", return_value=None):
                from backlog_synthesizer.tools.anthropic_generation import AnthropicGenerationTool

                with pytest.raises(PermanentToolError, match="No API key"):
                    AnthropicGenerationTool(api_key=None)
