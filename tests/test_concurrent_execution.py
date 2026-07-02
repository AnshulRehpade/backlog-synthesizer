"""Tests verifying concurrent execution within Parser and Story Writer agents.

Confirms that:
- Parser processes multiple documents concurrently via asyncio.gather()
- Story Writer generates multiple stories concurrently via asyncio.gather()
- Results are correct despite concurrent execution
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from backlog_synthesizer.agents.parser import ParserAgent
from backlog_synthesizer.agents.story_writer import StoryWriterAgent
from backlog_synthesizer.models.extraction import ExtractedItem, ExtractionResult
from backlog_synthesizer.models.gap_detection import GapReportEntry
from backlog_synthesizer.models.inputs import DocumentType, InputDocument
from backlog_synthesizer.models.output import AcceptanceCriterion, UserStory


# --- Parser concurrent document processing ---


class TestParserConcurrentDocuments:
    """Verify Parser processes multiple documents concurrently."""

    async def test_multiple_documents_processed_concurrently(self):
        """Multiple documents should be processed in parallel, not sequentially."""
        # Create a mock LLM tool that simulates a 0.1s delay per call
        call_times: list[float] = []

        def slow_generate(prompt, system_prompt=None):
            call_times.append(time.time())
            time.sleep(0.1)
            return '[]'  # Empty extraction

        mock_llm = MagicMock()
        mock_llm.generate = slow_generate

        mock_parser_tool = MagicMock()
        mock_parser_tool.chunk_text = MagicMock(return_value=["chunk1"])

        parser = ParserAgent(parsing_tool=mock_parser_tool, llm_tool=mock_llm)

        # Create 3 documents
        docs = [
            InputDocument(
                filename=f"doc{i}.txt",
                document_type=DocumentType.TRANSCRIPT_TXT,
                content=f"Meeting content for doc {i}".encode(),
                size_bytes=30,
            )
            for i in range(3)
        ]

        start = time.time()
        result = await parser.parse_documents(docs)
        elapsed = time.time() - start

        # If sequential, would take 3 * 0.1s = 0.3s minimum
        # If concurrent, should take ~0.1s (plus overhead)
        # We use a generous threshold to avoid flakiness
        assert elapsed < 0.25, f"Expected concurrent execution but took {elapsed:.2f}s"
        assert isinstance(result, ExtractionResult)

    async def test_single_document_still_works(self):
        """A single document should still work correctly."""
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value='[]')

        mock_parser_tool = MagicMock()

        parser = ParserAgent(parsing_tool=mock_parser_tool, llm_tool=mock_llm)

        docs = [
            InputDocument(
                filename="single.txt",
                document_type=DocumentType.TRANSCRIPT_TXT,
                content=b"Some meeting content",
                size_bytes=20,
            )
        ]

        result = await parser.parse_documents(docs)
        assert isinstance(result, ExtractionResult)
        assert result.errors == []

    async def test_error_in_one_document_doesnt_block_others(self):
        """An error parsing one document should not prevent others from completing."""
        def generate_items(prompt, system_prompt=None):
            return '[]'

        mock_llm = MagicMock()
        mock_llm.generate = generate_items

        # Make PDF parsing raise an error
        mock_parser_tool = MagicMock()
        mock_parser_tool.pdf_to_text = MagicMock(side_effect=Exception("Bad PDF"))

        parser = ParserAgent(parsing_tool=mock_parser_tool, llm_tool=mock_llm)

        docs = [
            InputDocument(
                filename="good.txt",
                document_type=DocumentType.TRANSCRIPT_TXT,
                content=b"Good content",
                size_bytes=12,
            ),
            InputDocument(
                filename="bad.pdf",
                document_type=DocumentType.TRANSCRIPT_PDF,
                content=b"not-a-pdf",
                size_bytes=9,
            ),
            InputDocument(
                filename="also_good.txt",
                document_type=DocumentType.TRANSCRIPT_TXT,
                content=b"Also good content",
                size_bytes=17,
            ),
        ]

        result = await parser.parse_documents(docs)
        # PDF produces an error, but txt docs still process
        assert isinstance(result, ExtractionResult)
        assert len(result.errors) >= 1
        assert result.errors[0].filename == "bad.pdf"

    async def test_concurrent_results_are_combined(self):
        """Items extracted from concurrent documents are combined into one result."""
        def generate_items(prompt, system_prompt=None):
            return '[{"item_type": "feature_request", "text": "A feature", "confidence": 0.9, "char_offset": 0, "stakeholder": null}]'

        mock_llm = MagicMock()
        mock_llm.generate = generate_items

        mock_parser_tool = MagicMock()

        parser = ParserAgent(parsing_tool=mock_parser_tool, llm_tool=mock_llm)

        docs = [
            InputDocument(
                filename=f"doc{i}.txt",
                document_type=DocumentType.TRANSCRIPT_TXT,
                content=b"Meeting content here",
                size_bytes=20,
            )
            for i in range(3)
        ]

        result = await parser.parse_documents(docs)
        # Each document produces 1 item, so we expect 3 total
        assert len(result.items) == 3
        assert all(item.item_type == "feature_request" for item in result.items)


# --- Story Writer concurrent story generation ---


class TestStoryWriterConcurrentGeneration:
    """Verify Story Writer generates stories for multiple items concurrently."""

    async def test_multiple_stories_generated_concurrently(self):
        """Multiple stories should be generated in parallel, not sequentially."""
        call_times: list[float] = []

        def slow_generate(prompt, system_prompt=None):
            call_times.append(time.time())
            time.sleep(0.1)
            return '''{
                "title": "Test Story",
                "user_story": "As a user, I want something, so that I benefit",
                "acceptance_criteria": [
                    {"description": "Criterion 1"},
                    {"description": "Criterion 2"}
                ],
                "tags": ["test"]
            }'''

        mock_llm = MagicMock()
        mock_llm.generate = slow_generate

        writer = StoryWriterAgent(llm_tool=mock_llm)

        # Create 4 gap report entries classified as "new"
        entries = [
            GapReportEntry(
                item=ExtractedItem(
                    item_type="feature_request",
                    text=f"Feature request number {i}",
                    source_chunk_index=0,
                    confidence=0.9,
                ),
                classification="new",
                gap_type="NEW",
                confidence=1.0,
                similarity_score=0.0,
            )
            for i in range(4)
        ]

        start = time.time()
        stories = await writer.generate_stories(entries)
        elapsed = time.time() - start

        # If sequential, would take 4 * 0.1s = 0.4s minimum
        # If concurrent, should take ~0.1s (plus overhead)
        assert elapsed < 0.25, f"Expected concurrent execution but took {elapsed:.2f}s"
        assert len(stories) == 4
        assert all(isinstance(s, UserStory) for s in stories)

    async def test_empty_entries_returns_empty(self):
        """Empty list of entries returns empty stories list."""
        mock_llm = MagicMock()
        writer = StoryWriterAgent(llm_tool=mock_llm)

        stories = await writer.generate_stories([])
        assert stories == []
        mock_llm.generate.assert_not_called()

    async def test_only_eligible_items_processed(self):
        """Only 'new' and 'conflict' items are processed; duplicates are skipped."""
        call_count = 0

        def counting_generate(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1
            return '''{
                "title": "Story",
                "user_story": "As a user, I want X, so that Y",
                "acceptance_criteria": [
                    {"description": "C1"},
                    {"description": "C2"}
                ],
                "tags": ["tag"]
            }'''

        mock_llm = MagicMock()
        mock_llm.generate = counting_generate

        writer = StoryWriterAgent(llm_tool=mock_llm)

        entries = [
            GapReportEntry(
                item=ExtractedItem(
                    item_type="feature_request",
                    text="New feature",
                    source_chunk_index=0,
                    confidence=0.9,
                ),
                classification="new",
                gap_type="NEW",
                confidence=1.0,
                similarity_score=0.0,
            ),
            GapReportEntry(
                item=ExtractedItem(
                    item_type="feature_request",
                    text="Duplicate feature",
                    source_chunk_index=0,
                    confidence=0.9,
                ),
                classification="duplicate",
                gap_type="DUPLICATE",
                confidence=0.9,
                similarity_score=0.9,
            ),
            GapReportEntry(
                item=ExtractedItem(
                    item_type="feature_request",
                    text="Conflicting feature",
                    source_chunk_index=0,
                    confidence=0.9,
                ),
                classification="conflict",
                gap_type="CONFLICT",
                confidence=0.7,
                similarity_score=0.7,
            ),
        ]

        stories = await writer.generate_stories(entries)
        # Only "new" and "conflict" are eligible — 2 items
        assert len(stories) == 2
        assert call_count == 2

    async def test_concurrent_stories_preserve_order(self):
        """Stories are returned in the same order as the input entries."""

        def ordered_generate(prompt, system_prompt=None):
            # Extract item text from prompt to create unique titles
            if "Feature A" in prompt:
                title = "Story A"
            elif "Feature B" in prompt:
                title = "Story B"
            elif "Feature C" in prompt:
                title = "Story C"
            else:
                title = "Unknown"

            return (
                '{"title": "' + title + '", '
                '"user_story": "As a user, I want X, so that Y", '
                '"acceptance_criteria": [{"description": "C1"}, {"description": "C2"}], '
                '"tags": ["tag"]}'
            )

        mock_llm = MagicMock()
        mock_llm.generate = ordered_generate

        writer = StoryWriterAgent(llm_tool=mock_llm)

        entries = [
            GapReportEntry(
                item=ExtractedItem(
                    item_type="feature_request",
                    text=f"Implement Feature {letter} with full detail",
                    source_chunk_index=i,
                    confidence=0.9,
                ),
                classification="new",
                gap_type="NEW",
                confidence=1.0,
                similarity_score=0.0,
            )
            for i, letter in enumerate(["A", "B", "C"])
        ]

        stories = await writer.generate_stories(entries)
        assert len(stories) == 3
        # asyncio.gather preserves order
        assert stories[0].title == "Story A"
        assert stories[1].title == "Story B"
        assert stories[2].title == "Story C"
