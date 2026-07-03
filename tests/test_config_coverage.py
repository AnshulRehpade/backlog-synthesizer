"""Unit tests for PipelineConfig with mocked environment and file I/O."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backlog_synthesizer.config import PipelineConfig


class TestPipelineConfigDefaults:
    def test_default_values(self):
        config = PipelineConfig()
        assert config.llm_provider == "anthropic"
        assert config.openai_model == "gpt-4o-mini"
        assert config.anthropic_model == "claude-haiku-4-20250414"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.chroma_collection == "backlog_items"
        assert config.react_reasoning_enabled is True
        assert config.otel_enabled is True


class TestPipelineConfigFromEnv:
    @patch.dict(
        "os.environ",
        {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "gpt-4",
            "OPENAI_BASE_URL": "https://custom.api",
            "OPENAI_TIMEOUT": "30",
            "ANTHROPIC_API_KEY": "ant-key",
            "ANTHROPIC_MODEL": "claude-3",
            "ANTHROPIC_MAX_TOKENS": "8192",
            "ANTHROPIC_TIMEOUT": "90",
            "EMBEDDING_MODEL": "custom-embed",
            "CHROMA_COLLECTION": "my_col",
            "CHROMA_PERSIST_DIR": "/tmp/chroma",
            "GAP_DETECTION_DUPLICATE_THRESHOLD": "0.9",
            "GAP_DETECTION_CONFLICT_THRESHOLD": "0.6",
            "REACT_REASONING_ENABLED": "false",
            "TOKENIZER_MODEL": "p50k_base",
            "OTEL_ENABLED": "no",
            "RUNS_DIR": "/tmp/runs",
            "EVAL_KEYWORD_THRESHOLD": "0.7",
            "EVAL_F1_THRESHOLD": "0.8",
            "EVAL_REGRESSION_THRESHOLD": "0.1",
            "FEW_SHOT_ENABLED": "false",
        },
        clear=True,
    )
    def test_from_env_reads_all_vars(self):
        config = PipelineConfig.from_env()
        assert config.llm_provider == "openai"
        assert config.openai_api_key == "sk-test"
        assert config.openai_model == "gpt-4"
        assert config.openai_base_url == "https://custom.api"
        assert config.openai_timeout == 30.0
        assert config.anthropic_api_key == "ant-key"
        assert config.anthropic_model == "claude-3"
        assert config.anthropic_max_tokens == 8192
        assert config.anthropic_timeout == 90.0
        assert config.embedding_model == "custom-embed"
        assert config.chroma_collection == "my_col"
        assert config.chroma_persist_dir == "/tmp/chroma"
        assert config.gap_detection_duplicate_threshold == 0.9
        assert config.gap_detection_conflict_threshold == 0.6
        assert config.react_reasoning_enabled is False
        assert config.tokenizer_model == "p50k_base"
        assert config.otel_enabled is False
        assert config.runs_dir == "/tmp/runs"
        assert config.few_shot_enabled is False

    @patch.dict("os.environ", {}, clear=True)
    def test_from_env_with_config_file(self, tmp_path):
        config_data = {"llm_provider": "openai", "openai_model": "gpt-4-turbo"}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = PipelineConfig.from_env(config_path=config_file)
        assert config.llm_provider == "openai"
        assert config.openai_model == "gpt-4-turbo"

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_config_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            PipelineConfig.from_env(config_path="/nonexistent/config.json")

    @patch.dict("os.environ", {}, clear=True)
    def test_invalid_json_raises(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text("not valid json {{{")

        with pytest.raises(ValueError, match="Invalid JSON"):
            PipelineConfig.from_env(config_path=config_file)


class TestPipelineConfigCreateLLMTool:
    @patch("backlog_synthesizer.tools.anthropic_generation.anthropic")
    def test_create_anthropic_tool(self, mock_anthropic):
        mock_anthropic.Anthropic.return_value = MagicMock()
        config = PipelineConfig(llm_provider="anthropic", anthropic_api_key="key")
        tool = config.create_llm_tool()
        assert tool.__class__.__name__ == "AnthropicGenerationTool"

    @patch("backlog_synthesizer.tools.llm_generation.openai")
    def test_create_openai_tool(self, mock_openai):
        mock_openai.OpenAI.return_value = MagicMock()
        config = PipelineConfig(llm_provider="openai", openai_api_key="key")
        tool = config.create_llm_tool()
        assert tool.__class__.__name__ == "OpenAIGenerationTool"

    def test_unknown_provider_raises(self):
        config = PipelineConfig(llm_provider="unknown")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            config.create_llm_tool()
