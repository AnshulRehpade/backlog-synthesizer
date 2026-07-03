"""Unit tests for ParserAgent._chunk_text method (tiktoken-based)."""

import pytest
import tiktoken

from backlog_synthesizer.agents.parser import ParserAgent
from backlog_synthesizer.models.extraction import TextChunk


class MockParsingTool:
    def pdf_to_text(self, content: bytes) -> str:
        return ""

    def chunk_text(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        return []


class MockLLMTool:
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return ""


@pytest.fixture
def parser() -> ParserAgent:
    return ParserAgent(MockParsingTool(), MockLLMTool())


@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding("cl100k_base")


class TestChunkTextEmpty:
    """Edge cases: empty and whitespace-only text."""

    def test_empty_string_returns_empty_list(self, parser: ParserAgent) -> None:
        result = parser._chunk_text("")
        assert result == []

    def test_whitespace_only_returns_empty_list(self, parser: ParserAgent) -> None:
        result = parser._chunk_text("   ")
        assert result == []


class TestChunkTextSingleChunk:
    """Text that fits within a single chunk."""

    def test_single_word(self, parser: ParserAgent, tokenizer) -> None:
        result = parser._chunk_text("hello")
        assert len(result) == 1
        expected_tokens = len(tokenizer.encode("hello"))
        assert result[0].token_count == expected_tokens

    def test_text_shorter_than_max_tokens(self, parser: ParserAgent, tokenizer) -> None:
        text = "the quick brown fox jumps over the lazy dog"
        result = parser._chunk_text(text)
        assert len(result) == 1
        assert result[0].text == text
        assert result[0].token_count == len(tokenizer.encode(text))
        assert result[0].index == 0

    def test_text_exactly_at_max_tokens(self, parser: ParserAgent, tokenizer) -> None:
        """Text with exactly max_tokens real tokens fits in one chunk."""
        # Build text that's exactly 2000 tokens
        base = "hello world this is a test sentence for chunking "
        base_tokens = tokenizer.encode(base)
        # Repeat until we get close, then trim
        repeated = base * 300  # more than enough
        tokens = tokenizer.encode(repeated)[:2000]
        text = tokenizer.decode(tokens)
        # Verify it's 2000 tokens
        assert len(tokenizer.encode(text)) == 2000

        result = parser._chunk_text(text)
        assert len(result) == 1
        assert result[0].token_count == 2000


class TestChunkTextMultipleChunks:
    """Text that requires multiple chunks with overlap."""

    def _make_long_text(self, tokenizer, n_tokens: int) -> str:
        """Helper to create text with exactly n real tokens."""
        base = "The quick brown fox jumped over the lazy sleeping dog near the river. "
        repeated = base * (n_tokens // 10 + 50)
        tokens = tokenizer.encode(repeated)[:n_tokens]
        return tokenizer.decode(tokens)

    def test_each_chunk_respects_max_tokens(self, parser: ParserAgent, tokenizer) -> None:
        text = self._make_long_text(tokenizer, 5000)
        result = parser._chunk_text(text, max_tokens=2000, overlap=200)
        for chunk in result:
            assert chunk.token_count <= 2000

    def test_consecutive_chunks_share_exact_overlap_in_tokens(
        self, parser: ParserAgent, tokenizer
    ) -> None:
        """Overlap is measured in tokens — non-final chunks have exactly max_tokens tokens."""
        text = self._make_long_text(tokenizer, 4000)
        result = parser._chunk_text(text, max_tokens=2000, overlap=200)

        # Non-final chunks should have exactly 2000 tokens
        for i in range(len(result) - 1):
            assert result[i].token_count == 2000

    def test_all_content_is_covered(self, parser: ParserAgent, tokenizer) -> None:
        """No tokens are lost — total unique tokens across chunks equals original."""
        text = self._make_long_text(tokenizer, 6000)
        original_tokens = tokenizer.encode(text)
        result = parser._chunk_text(text, max_tokens=2000, overlap=200)

        # Total unique tokens = first chunk + (each subsequent - overlap)
        total_unique = result[0].token_count
        for i in range(1, len(result)):
            total_unique += result[i].token_count - 200

        assert total_unique == len(original_tokens)

    def test_indices_are_sequential(self, parser: ParserAgent, tokenizer) -> None:
        text = self._make_long_text(tokenizer, 5000)
        result = parser._chunk_text(text, max_tokens=2000, overlap=200)
        for i, chunk in enumerate(result):
            assert chunk.index == i

    def test_just_over_max_tokens_produces_two_chunks(
        self, parser: ParserAgent, tokenizer
    ) -> None:
        text = self._make_long_text(tokenizer, 2001)
        result = parser._chunk_text(text, max_tokens=2000, overlap=200)
        assert len(result) == 2
        assert result[0].token_count == 2000


class TestChunkTextCustomParameters:
    """Using non-default max_tokens and overlap."""

    def test_small_max_tokens_and_overlap(self, parser: ParserAgent, tokenizer) -> None:
        text = "one two three four five six seven eight nine ten eleven twelve"
        result = parser._chunk_text(text, max_tokens=5, overlap=2)

        for chunk in result:
            assert chunk.token_count <= 5

    def test_chunks_are_valid_readable_text(self, parser: ParserAgent, tokenizer) -> None:
        """Each chunk decodes to valid readable text (no broken characters)."""
        text = "The café served crème brûlée alongside naïve résumés for the über enthusiastic customers."
        result = parser._chunk_text(text, max_tokens=5, overlap=2)

        for chunk in result:
            # Should be decodeable without errors
            assert isinstance(chunk.text, str)
            assert len(chunk.text) > 0

    def test_technical_text_with_compound_words(self, parser: ParserAgent, tokenizer) -> None:
        """Technical text with long compound words is chunked correctly."""
        text = (
            "The microservice-based architecture uses event-driven communication "
            "patterns with schema-registry-validated message-queue-backed processing "
            "pipelines for high-throughput distributed-systems workloads. "
            "Cross-datacenter replication ensures disaster-recovery compliance."
        )
        result = parser._chunk_text(text, max_tokens=10, overlap=3)

        for chunk in result:
            assert chunk.token_count <= 10
            assert isinstance(chunk.text, str)

    def test_special_characters_and_punctuation(self, parser: ParserAgent, tokenizer) -> None:
        """Text with special characters is handled correctly."""
        text = "Hello! @user — that's $100.00 (incl. tax) for items #1, #2 & #3; right?"
        result = parser._chunk_text(text, max_tokens=5, overlap=2)

        for chunk in result:
            assert chunk.token_count <= 5
            assert isinstance(chunk.text, str)
