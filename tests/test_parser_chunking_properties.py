"""Property-based tests for ParserAgent._chunk_text method.

Uses Hypothesis to verify that document chunking preserves content with bounded size.
"""

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


# --- Strategies ---

# Strategy for generating text with whitespace-delimited tokens
# Uses printable characters without whitespace to form tokens, joined by spaces
token_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S"), blacklist_characters=" \t\n\r"),
    min_size=1,
    max_size=20,
).filter(lambda t: len(t.strip()) > 0)

# Strategy for generating text with many tokens (to exercise multi-chunk paths)
multi_token_text = st.lists(token_strategy, min_size=1, max_size=6000).map(lambda tokens: " ".join(tokens))

# Strategy for generating short text (single chunk)
short_text = st.lists(token_strategy, min_size=1, max_size=50).map(lambda tokens: " ".join(tokens))

# Strategy for custom chunking parameters
custom_max_tokens = st.integers(min_value=10, max_value=500)
custom_overlap = st.integers(min_value=1, max_value=50)


# --- Property Tests ---


# Feature: backlog-synthesizer, Property 2: Document chunking preserves content with bounded size
class TestProperty2DocumentChunkingDefaultParams:
    """Verify document chunking preserves content with bounded size using default parameters."""

    @given(text=multi_token_text)
    @settings(max_examples=100)
    def test_chunk_size_bounded_by_max_tokens(self, text: str) -> None:
        """
        For any input text, each chunk produced by _chunk_text with max_tokens=2000
        contains at most 2000 tokens.

        **Validates: Requirements 3.5**
        """
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        chunks = parser._chunk_text(text, max_tokens=2000, overlap=200)

        for chunk in chunks:
            token_count = len(chunk.text.split())
            assert token_count <= 2000, (
                f"Chunk {chunk.index} has {token_count} tokens, exceeding max_tokens=2000"
            )

    @given(text=multi_token_text)
    @settings(max_examples=100)
    def test_consecutive_chunks_share_exact_overlap(self, text: str) -> None:
        """
        For any input text producing multiple chunks, consecutive chunks share
        exactly 200 tokens of overlap (last 200 tokens of chunk[i] equal first
        200 tokens of chunk[i+1]).

        **Validates: Requirements 3.5**
        """
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        chunks = parser._chunk_text(text, max_tokens=2000, overlap=200)

        if len(chunks) <= 1:
            return  # Overlap only applies with multiple chunks

        for i in range(len(chunks) - 1):
            a_tokens = chunks[i].text.split()
            b_tokens = chunks[i + 1].text.split()
            assert a_tokens[-200:] == b_tokens[:200], (
                f"Chunks {i} and {i+1} do not share exactly 200 tokens of overlap"
            )

    @given(text=multi_token_text)
    @settings(max_examples=100)
    def test_reconstruction_reproduces_original_text(self, text: str) -> None:
        """
        For any input text, reconstructing from chunks by removing overlapping
        regions reproduces the original text exactly.

        **Validates: Requirements 3.5**
        """
        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        chunks = parser._chunk_text(text, max_tokens=2000, overlap=200)

        if not chunks:
            # Empty text produces no chunks; original must be whitespace-only
            assert text.split() == []
            return

        # Reconstruct: take all tokens from first chunk, then non-overlapping
        # tokens from subsequent chunks
        all_tokens = chunks[0].text.split()
        for i in range(1, len(chunks)):
            chunk_tokens = chunks[i].text.split()
            all_tokens.extend(chunk_tokens[200:])

        reconstructed = " ".join(all_tokens)
        original_normalized = " ".join(text.split())
        assert reconstructed == original_normalized, (
            f"Reconstruction does not match original text. "
            f"Original tokens: {len(original_normalized.split())}, "
            f"Reconstructed tokens: {len(reconstructed.split())}"
        )


# Feature: backlog-synthesizer, Property 2: Document chunking preserves content with bounded size
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
        each chunk contains at most max_tokens tokens.

        **Validates: Requirements 3.5**
        """
        # Overlap must be less than max_tokens for valid chunking
        if overlap >= max_tokens:
            return

        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        chunks = parser._chunk_text(text, max_tokens=max_tokens, overlap=overlap)

        for chunk in chunks:
            token_count = len(chunk.text.split())
            assert token_count <= max_tokens, (
                f"Chunk {chunk.index} has {token_count} tokens, "
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
        consecutive chunks share exactly `overlap` tokens.

        **Validates: Requirements 3.5**
        """
        if overlap >= max_tokens:
            return

        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        chunks = parser._chunk_text(text, max_tokens=max_tokens, overlap=overlap)

        if len(chunks) <= 1:
            return

        for i in range(len(chunks) - 1):
            a_tokens = chunks[i].text.split()
            b_tokens = chunks[i + 1].text.split()
            # The last chunk may have fewer tokens than overlap if it's the tail
            # But for non-final chunks, they should have max_tokens tokens
            assert a_tokens[-overlap:] == b_tokens[:overlap], (
                f"Chunks {i} and {i+1} do not share exactly {overlap} tokens of overlap"
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
        reconstructing from chunks reproduces the original text.

        **Validates: Requirements 3.5**
        """
        if overlap >= max_tokens:
            return

        parser = ParserAgent(MockParsingTool(), MockLLMTool())
        chunks = parser._chunk_text(text, max_tokens=max_tokens, overlap=overlap)

        if not chunks:
            assert text.split() == []
            return

        all_tokens = chunks[0].text.split()
        for i in range(1, len(chunks)):
            chunk_tokens = chunks[i].text.split()
            all_tokens.extend(chunk_tokens[overlap:])

        reconstructed = " ".join(all_tokens)
        original_normalized = " ".join(text.split())
        assert reconstructed == original_normalized, (
            f"Reconstruction does not match original. "
            f"max_tokens={max_tokens}, overlap={overlap}"
        )
