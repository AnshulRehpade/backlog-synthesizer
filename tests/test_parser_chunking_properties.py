"""Property-based tests for ParserAgent._chunk_text method (tiktoken-based).

Uses Hypothesis to verify that document chunking preserves content with bounded size.
Token counting uses tiktoken (cl100k_base) to match the implementation.
"""

import tiktoken
from hypothesis import given, settings
from hypothesis import strategies as st

from backlog_synthesizer.agents.parser import ParserAgent
from backlog_synthesizer.models.extraction import TextChunk


# --- Mock Tools ---


class MockParsingTool:
    def pdf_to_text(self, content: bytes) -> str:
        return ""

    def chunk_text(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        return []


class MockLLMTool:
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return ""


# Shared tokenizer for test verification
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# --- Strategies ---

# Strategy for generating text with varied content
token_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S"), blacklist_characters="\x00"),
    min_size=1,
    max_size=20,
).filter(lambda t: len(t.strip()) > 0)

# Multi-token text (generates long enough text to exercise multi-chunk paths)
multi_token_text = st.lists(token_strategy, min_size=10, max_size=500).map(
    lambda tokens: " ".join(tokens)
)

# Strategy for custom chunking parameters
custom_max_tokens = st.integers(min_value=20, max_value=200)
custom_overlap = st.integers(min_value=2, max_value=15)


# --- Property Tests ---


# Feature: backlog-synthesizer, Property 2: Document chunking preserves content with bounded size
class TestProperty2DocumentChunkingDefaultParams:
    """Verify document chunking preserves content with bounded size using default parameters."""

    @given(text=multi_token_text)
    @settings(max_examples=100)
    def test_chunk_size_bounded_by_max_tokens(self, text: str) -> None:
        """
        For any input text, each chunk produced by _chunk_text with max_tokens=2000
        contains at most 2000 real tokens.

        **Validates: Requirements 3.5**
        """
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        chunks = parser._chunk_text(text, max_tokens=2000, overlap=200)

        for chunk in chunks:
            actual_tokens = len(_TOKENIZER.encode(chunk.text))
            assert actual_tokens <= 2000, (
                f"Chunk {chunk.index} has {actual_tokens} tokens, exceeding max_tokens=2000"
            )

    @given(text=multi_token_text)
    @settings(max_examples=100)
    def test_token_count_field_is_accurate(self, text: str) -> None:
        """
        For any input text, each chunk's token_count field matches the actual
        token count via tiktoken.

        **Validates: Requirements 3.5**
        """
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        chunks = parser._chunk_text(text, max_tokens=2000, overlap=200)

        for chunk in chunks:
            actual = len(_TOKENIZER.encode(chunk.text))
            assert chunk.token_count == actual, (
                f"Chunk {chunk.index}: token_count={chunk.token_count} but actual={actual}"
            )

    @given(text=multi_token_text)
    @settings(max_examples=100)
    def test_consecutive_chunks_share_exact_overlap_in_tokens(self, text: str) -> None:
        """
        For any input text producing multiple chunks, non-final chunks have
        exactly max_tokens (2000) tokens, ensuring the overlap step is correct.

        **Validates: Requirements 3.5**
        """
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        chunks = parser._chunk_text(text, max_tokens=2000, overlap=200)

        if len(chunks) <= 1:
            return

        # Non-final chunks should have exactly max_tokens tokens
        for i in range(len(chunks) - 1):
            assert chunks[i].token_count == 2000, (
                f"Non-final chunk {i} should have exactly 2000 tokens, got {chunks[i].token_count}"
            )

    @given(text=multi_token_text)
    @settings(max_examples=100)
    def test_reconstruction_reproduces_original_tokens(self, text: str) -> None:
        """
        For any input text, the total tokens across all chunks (accounting for
        overlap) equals the original token count.

        **Validates: Requirements 3.5**
        """
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        chunks = parser._chunk_text(text, max_tokens=2000, overlap=200)

        if not chunks:
            assert not text.strip()
            return

        original_count = len(_TOKENIZER.encode(text))

        # Total unique tokens = first chunk + (each subsequent chunk - overlap)
        total_unique = chunks[0].token_count
        for i in range(1, len(chunks)):
            total_unique += chunks[i].token_count - 200

        assert total_unique == original_count, (
            f"Token coverage mismatch: original={original_count}, reconstructed={total_unique}"
        )


class TestProperty2DocumentChunkingCustomParams:
    """Verify document chunking properties hold with custom max_tokens and overlap."""

    @given(
        text=multi_token_text,
        max_tokens=custom_max_tokens,
        overlap=custom_overlap,
    )
    @settings(max_examples=100)
    def test_chunk_size_bounded_custom_params(
        self, text: str, max_tokens: int, overlap: int
    ) -> None:
        """
        For any input text and valid custom parameters where overlap < max_tokens,
        each chunk contains at most max_tokens real tokens.

        **Validates: Requirements 3.5**
        """
        if overlap >= max_tokens:
            return

        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        chunks = parser._chunk_text(text, max_tokens=max_tokens, overlap=overlap)

        for chunk in chunks:
            actual_tokens = len(_TOKENIZER.encode(chunk.text))
            assert actual_tokens <= max_tokens, (
                f"Chunk {chunk.index} has {actual_tokens} tokens, "
                f"exceeding max_tokens={max_tokens}"
            )

    @given(
        text=multi_token_text,
        max_tokens=custom_max_tokens,
        overlap=custom_overlap,
    )
    @settings(max_examples=100)
    def test_overlap_exact_custom_params(
        self, text: str, max_tokens: int, overlap: int
    ) -> None:
        """
        For any input text and valid custom parameters where overlap < max_tokens,
        the chunk step size is (max_tokens - overlap), ensuring overlap coverage.

        **Validates: Requirements 3.5**
        """
        if overlap >= max_tokens:
            return

        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        chunks = parser._chunk_text(text, max_tokens=max_tokens, overlap=overlap)

        if len(chunks) <= 1:
            return

        # Verify overlap property: for non-final chunks, token_count == max_tokens
        # This means they advance by (max_tokens - overlap) each step
        for i in range(len(chunks) - 1):
            assert chunks[i].token_count == max_tokens, (
                f"Non-final chunk {i} should have exactly max_tokens={max_tokens} tokens, "
                f"got {chunks[i].token_count}"
            )

    @given(
        text=multi_token_text,
        max_tokens=custom_max_tokens,
        overlap=custom_overlap,
    )
    @settings(max_examples=100)
    def test_reconstruction_custom_params(
        self, text: str, max_tokens: int, overlap: int
    ) -> None:
        """
        For any input text and valid custom parameters where overlap < max_tokens,
        the total unique token count across chunks equals the original.

        **Validates: Requirements 3.5**
        """
        if overlap >= max_tokens:
            return

        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        chunks = parser._chunk_text(text, max_tokens=max_tokens, overlap=overlap)

        if not chunks:
            assert not text.strip()
            return

        original_count = len(_TOKENIZER.encode(text))

        total_unique = chunks[0].token_count
        for i in range(1, len(chunks)):
            total_unique += chunks[i].token_count - overlap

        assert total_unique == original_count, (
            f"Token coverage mismatch: original={original_count}, reconstructed={total_unique}. "
            f"max_tokens={max_tokens}, overlap={overlap}"
        )
