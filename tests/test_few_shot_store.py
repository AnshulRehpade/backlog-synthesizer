"""Tests for the FewShotStore — indexing and retrieval of few-shot examples."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backlog_synthesizer.evaluation.few_shot_store import (
    FewShotExample,
    FewShotStore,
    SIMILARITY_THRESHOLD,
)
from backlog_synthesizer.tools.interfaces import SearchResult


class TestFewShotStoreIndexing:
    def test_index_golden_dataset_returns_count(self, tmp_path):
        """Indexing returns the number of entries indexed."""
        entries = [
            {"id": "test_001", "transcript": "Add login feature", "tags": ["auth"]},
            {"id": "test_002", "transcript": "Fix bug in payment", "tags": ["bug"]},
        ]
        dataset_file = tmp_path / "golden_entries.json"
        dataset_file.write_text(json.dumps(entries))

        mock_embedding = MagicMock()
        mock_embedding.generate_embedding = MagicMock(return_value=[0.1] * 384)
        mock_vector = MagicMock()
        mock_vector.store = MagicMock()

        store = FewShotStore(mock_embedding, mock_vector)
        count = store.index_golden_dataset(dataset_file)

        assert count == 2
        assert mock_vector.store.call_count == 2

    def test_index_empty_dataset(self, tmp_path):
        """Indexing empty dataset returns 0."""
        dataset_file = tmp_path / "golden_entries.json"
        dataset_file.write_text("[]")

        mock_embedding = MagicMock()
        mock_vector = MagicMock()

        store = FewShotStore(mock_embedding, mock_vector)
        count = store.index_golden_dataset(dataset_file)
        assert count == 0

    def test_index_missing_file(self):
        """Missing dataset file returns 0 without error."""
        mock_embedding = MagicMock()
        mock_vector = MagicMock()

        store = FewShotStore(mock_embedding, mock_vector)
        count = store.index_golden_dataset(Path("/nonexistent/path.json"))
        assert count == 0

    def test_index_invalid_json(self, tmp_path):
        """Invalid JSON in dataset file returns 0 without error."""
        dataset_file = tmp_path / "golden_entries.json"
        dataset_file.write_text("not valid json{{{")

        mock_embedding = MagicMock()
        mock_vector = MagicMock()

        store = FewShotStore(mock_embedding, mock_vector)
        count = store.index_golden_dataset(dataset_file)
        assert count == 0

    def test_index_skips_entries_without_transcript(self, tmp_path):
        """Entries without transcript or id are skipped."""
        entries = [
            {"id": "test_001", "transcript": "", "tags": ["auth"]},  # empty transcript
            {"id": "", "transcript": "Some text", "tags": ["bug"]},  # empty id
            {"id": "test_003", "transcript": "Valid entry", "tags": ["feature"]},
        ]
        dataset_file = tmp_path / "golden_entries.json"
        dataset_file.write_text(json.dumps(entries))

        mock_embedding = MagicMock()
        mock_embedding.generate_embedding = MagicMock(return_value=[0.1] * 384)
        mock_vector = MagicMock()
        mock_vector.store = MagicMock()

        store = FewShotStore(mock_embedding, mock_vector)
        count = store.index_golden_dataset(dataset_file)
        assert count == 1

    def test_index_continues_on_embedding_error(self, tmp_path):
        """Embedding errors for individual entries don't stop indexing."""
        entries = [
            {"id": "test_001", "transcript": "First entry", "tags": []},
            {"id": "test_002", "transcript": "Second entry", "tags": []},
        ]
        dataset_file = tmp_path / "golden_entries.json"
        dataset_file.write_text(json.dumps(entries))

        mock_embedding = MagicMock()
        mock_embedding.generate_embedding = MagicMock(
            side_effect=[RuntimeError("fail"), [0.1] * 384]
        )
        mock_vector = MagicMock()
        mock_vector.store = MagicMock()

        store = FewShotStore(mock_embedding, mock_vector)
        count = store.index_golden_dataset(dataset_file)
        assert count == 1


class TestFewShotStoreRetrieval:
    def test_get_similar_examples_returns_above_threshold(self, tmp_path):
        """Only returns examples with score >= threshold."""
        entries = [
            {"id": "g001", "transcript": "Add auth", "description": "Auth feature", "tags": ["auth"]},
            {"id": "g002", "transcript": "Fix bug", "description": "Bug fix", "tags": ["bug"]},
        ]
        dataset_file = tmp_path / "golden_entries.json"
        dataset_file.write_text(json.dumps(entries))

        mock_embedding = MagicMock()
        mock_embedding.generate_embedding = MagicMock(return_value=[0.1] * 384)
        mock_vector = MagicMock()
        mock_vector.store = MagicMock()
        mock_vector.query_similar = MagicMock(
            return_value=[
                SearchResult(item_id="few_shot_g001", score=0.85, metadata={"golden_id": "g001"}),
                SearchResult(item_id="few_shot_g002", score=0.3, metadata={"golden_id": "g002"}),
            ]
        )

        store = FewShotStore(mock_embedding, mock_vector)
        store.index_golden_dataset(dataset_file)

        examples = store.get_similar_parser_examples("authentication feature")
        assert len(examples) == 1
        assert examples[0].id == "g001"
        assert examples[0].score == 0.85

    def test_returns_empty_when_not_indexed(self):
        """Returns empty list if not indexed."""
        mock_embedding = MagicMock()
        mock_vector = MagicMock()

        store = FewShotStore(mock_embedding, mock_vector)
        examples = store.get_similar_parser_examples("some text")
        assert examples == []

    def test_returns_empty_on_retrieval_error(self, tmp_path):
        """Returns empty list if retrieval raises an exception."""
        entries = [{"id": "g001", "transcript": "Test", "tags": []}]
        dataset_file = tmp_path / "golden_entries.json"
        dataset_file.write_text(json.dumps(entries))

        mock_embedding = MagicMock()
        mock_embedding.generate_embedding = MagicMock(
            side_effect=[
                [0.1] * 384,  # for indexing
                RuntimeError("embedding failed"),  # for query
            ]
        )
        mock_vector = MagicMock()
        mock_vector.store = MagicMock()

        store = FewShotStore(mock_embedding, mock_vector)
        store.index_golden_dataset(dataset_file)

        examples = store.get_similar_parser_examples("query text")
        assert examples == []

    def test_get_similar_story_examples_same_as_parser(self, tmp_path):
        """Story examples use the same retrieval logic."""
        entries = [
            {"id": "g001", "transcript": "Build dashboard", "description": "Dashboard", "tags": ["ui"]}
        ]
        dataset_file = tmp_path / "golden_entries.json"
        dataset_file.write_text(json.dumps(entries))

        mock_embedding = MagicMock()
        mock_embedding.generate_embedding = MagicMock(return_value=[0.1] * 384)
        mock_vector = MagicMock()
        mock_vector.store = MagicMock()
        mock_vector.query_similar = MagicMock(
            return_value=[
                SearchResult(item_id="few_shot_g001", score=0.75, metadata={"golden_id": "g001"}),
            ]
        )

        store = FewShotStore(mock_embedding, mock_vector)
        store.index_golden_dataset(dataset_file)

        examples = store.get_similar_story_examples("Create user dashboard")
        assert len(examples) == 1

    def test_respects_top_k_parameter(self, tmp_path):
        """Returns at most top_k results."""
        entries = [
            {"id": f"g{i:03d}", "transcript": f"Entry {i}", "description": f"Desc {i}", "tags": []}
            for i in range(5)
        ]
        dataset_file = tmp_path / "golden_entries.json"
        dataset_file.write_text(json.dumps(entries))

        mock_embedding = MagicMock()
        mock_embedding.generate_embedding = MagicMock(return_value=[0.1] * 384)
        mock_vector = MagicMock()
        mock_vector.store = MagicMock()
        mock_vector.query_similar = MagicMock(
            return_value=[
                SearchResult(item_id=f"few_shot_g{i:03d}", score=0.9 - i * 0.05, metadata={"golden_id": f"g{i:03d}"})
                for i in range(3)
            ]
        )

        store = FewShotStore(mock_embedding, mock_vector)
        store.index_golden_dataset(dataset_file)

        examples = store.get_similar_parser_examples("test", top_k=3)
        assert len(examples) == 3

    def test_all_below_threshold_returns_empty(self, tmp_path):
        """Returns empty list when all results are below threshold."""
        entries = [{"id": "g001", "transcript": "Irrelevant", "description": "X", "tags": []}]
        dataset_file = tmp_path / "golden_entries.json"
        dataset_file.write_text(json.dumps(entries))

        mock_embedding = MagicMock()
        mock_embedding.generate_embedding = MagicMock(return_value=[0.1] * 384)
        mock_vector = MagicMock()
        mock_vector.store = MagicMock()
        mock_vector.query_similar = MagicMock(
            return_value=[
                SearchResult(item_id="few_shot_g001", score=0.2, metadata={"golden_id": "g001"}),
            ]
        )

        store = FewShotStore(mock_embedding, mock_vector)
        store.index_golden_dataset(dataset_file)

        examples = store.get_similar_parser_examples("unrelated query")
        assert examples == []


class TestFewShotFallback:
    def test_parser_works_without_few_shot_store(self):
        """Parser works normally when few_shot_store=None."""
        from backlog_synthesizer.agents.parser import ParserAgent

        class MockParsingTool:
            def pdf_to_text(self, content):
                return ""

            def chunk_text(self, text, max_tokens, overlap):
                return []

        class MockLLMTool:
            def generate(self, prompt, system_prompt=None):
                return "[]"

        parser = ParserAgent(MockParsingTool(), MockLLMTool(), few_shot_store=None)
        # Should not raise
        assert parser._few_shot_store is None

    def test_story_writer_works_without_few_shot_store(self):
        """StoryWriter works normally when few_shot_store=None."""
        from backlog_synthesizer.agents.story_writer import StoryWriterAgent

        class MockLLMTool:
            def generate(self, prompt, system_prompt=None):
                return "{}"

        writer = StoryWriterAgent(MockLLMTool(), few_shot_store=None)
        assert writer._few_shot_store is None

    def test_parser_with_few_shot_store(self):
        """Parser accepts a few_shot_store parameter."""
        from backlog_synthesizer.agents.parser import ParserAgent

        class MockParsingTool:
            def pdf_to_text(self, content):
                return ""

            def chunk_text(self, text, max_tokens, overlap):
                return []

        class MockLLMTool:
            def generate(self, prompt, system_prompt=None):
                return "[]"

        mock_store = MagicMock()
        parser = ParserAgent(MockParsingTool(), MockLLMTool(), few_shot_store=mock_store)
        assert parser._few_shot_store is mock_store

    def test_story_writer_with_few_shot_store(self):
        """StoryWriter accepts a few_shot_store parameter."""
        from backlog_synthesizer.agents.story_writer import StoryWriterAgent

        class MockLLMTool:
            def generate(self, prompt, system_prompt=None):
                return "{}"

        mock_store = MagicMock()
        writer = StoryWriterAgent(MockLLMTool(), few_shot_store=mock_store)
        assert writer._few_shot_store is mock_store


class TestFewShotConfig:
    def test_config_has_few_shot_enabled_default_true(self):
        """PipelineConfig defaults few_shot_enabled to True."""
        from backlog_synthesizer.config import PipelineConfig

        config = PipelineConfig()
        assert config.few_shot_enabled is True

    def test_config_few_shot_disabled(self):
        """PipelineConfig can disable few-shot."""
        from backlog_synthesizer.config import PipelineConfig

        config = PipelineConfig(few_shot_enabled=False)
        assert config.few_shot_enabled is False
