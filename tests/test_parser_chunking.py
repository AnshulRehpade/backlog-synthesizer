"""Unit tests for ParserAgent._chunk_text method."""

import pytest

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

    def test_single_word(self, parser: ParserAgent) -> None:
        result = parser._chunk_text("hello")
        assert len(result) == 1
        assert result[0] == TextChunk(index=0, text="hello", token_count=1)

    def test_text_shorter_than_max_tokens(self, parser: ParserAgent) -> None:
        text = "the quick brown fox jumps over the lazy dog"
        result = parser._chunk_text(text)
        assert len(result) == 1
        assert result[0].text == text
        assert result[0].token_count == 9
        assert result[0].index == 0

    def test_text_exactly_at_max_tokens(self, parser: ParserAgent) -> None:
        tokens = [f"word{i}" for i in range(2000)]
        text = " ".join(tokens)
        result = parser._chunk_text(text)
        assert len(result) == 1
        assert result[0].token_count == 2000


class TestChunkTextMultipleChunks:
    """Text that requires multiple chunks with overlap."""

    def test_each_chunk_respects_max_tokens(self, parser: ParserAgent) -> None:
        tokens = [f"w{i}" for i in range(5000)]
        text = " ".join(tokens)
        result = parser._chunk_text(text, max_tokens=2000, overlap=200)
        for chunk in result:
            assert chunk.token_count <= 2000

    def test_consecutive_chunks_share_exact_overlap(self, parser: ParserAgent) -> None:
        tokens = [f"t{i}" for i in range(4000)]
        text = " ".join(tokens)
        result = parser._chunk_text(text, max_tokens=2000, overlap=200)
        for i in range(len(result) - 1):
            a_tokens = result[i].text.split()
            b_tokens = result[i + 1].text.split()
            assert a_tokens[-200:] == b_tokens[:200]

    def test_reconstruction_reproduces_original(self, parser: ParserAgent) -> None:
        tokens = [f"x{i}" for i in range(6000)]
        text = " ".join(tokens)
        result = parser._chunk_text(text, max_tokens=2000, overlap=200)

        # Reconstruct by removing overlap from subsequent chunks
        all_tokens = result[0].text.split()
        for i in range(1, len(result)):
            chunk_tokens = result[i].text.split()
            all_tokens.extend(chunk_tokens[200:])

        reconstructed = " ".join(all_tokens)
        assert reconstructed == text

    def test_indices_are_sequential(self, parser: ParserAgent) -> None:
        tokens = [f"a{i}" for i in range(5000)]
        text = " ".join(tokens)
        result = parser._chunk_text(text, max_tokens=2000, overlap=200)
        for i, chunk in enumerate(result):
            assert chunk.index == i

    def test_just_over_max_tokens_produces_two_chunks(self, parser: ParserAgent) -> None:
        tokens = [f"b{i}" for i in range(2001)]
        text = " ".join(tokens)
        result = parser._chunk_text(text, max_tokens=2000, overlap=200)
        assert len(result) == 2
        assert result[0].token_count == 2000
        assert result[1].token_count == 201


class TestChunkTextCustomParameters:
    """Using non-default max_tokens and overlap."""

    def test_small_max_tokens_and_overlap(self, parser: ParserAgent) -> None:
        tokens = [f"t{i}" for i in range(20)]
        text = " ".join(tokens)
        result = parser._chunk_text(text, max_tokens=5, overlap=2)

        for chunk in result:
            assert chunk.token_count <= 5

        for i in range(len(result) - 1):
            a = result[i].text.split()
            b = result[i + 1].text.split()
            assert a[-2:] == b[:2]

        # Verify reconstruction
        all_tokens = result[0].text.split()
        for i in range(1, len(result)):
            chunk_tokens = result[i].text.split()
            all_tokens.extend(chunk_tokens[2:])
        assert " ".join(all_tokens) == text
