"""Unit tests for the StoryWriterAgent class.

Tests the agent's generate_stories method with mocked LLM tool
to verify user story generation, parsing, and edge case handling.

Requirements: 5.1, 5.2, 5.3, 5.6
"""

import json

import pytest

from backlog_synthesizer.agents.story_writer import (
    ELIGIBLE_CLASSIFICATIONS,
    StoryWriterAgent,
    _create_placeholder_story,
    _has_insufficient_detail,
    _parse_llm_response,
)
from backlog_synthesizer.models.extraction import ExtractedItem
from backlog_synthesizer.models.gap_detection import GapReportEntry


# --- Mock Tool Implementations ---


class MockLLMGenerationTool:
    """Mock LLM tool with configurable response."""

    def __init__(self, response: str | None = None):
        self._response = response or json.dumps({
            "title": "User Authentication",
            "user_story": "As a user, I want to log in securely, so that my data is protected",
            "acceptance_criteria": [
                {"description": "User can enter email and password"},
                {"description": "System validates credentials against stored data"},
                {"description": "User receives error message for invalid credentials"},
            ],
            "tags": ["authentication", "security"],
        })

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self._response


class FailingLLMTool:
    """LLM tool that raises an exception."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        raise RuntimeError("LLM service unavailable")


class StatefulLLMTool:
    """LLM tool that returns different responses per call."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._call_count = 0

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        idx = self._call_count
        self._call_count += 1
        if idx < len(self._responses):
            return self._responses[idx]
        return self._responses[-1]


# --- Helper ---


def make_item(
    text: str = "Users need a way to export their data in CSV format",
    item_type: str = "feature_request",
    confidence: float = 0.9,
    stakeholder: str | None = "Product Manager",
    section_heading: str | None = None,
) -> ExtractedItem:
    return ExtractedItem(
        item_type=item_type,
        text=text,
        source_chunk_index=0,
        confidence=confidence,
        stakeholder=stakeholder,
        section_heading=section_heading,
    )


def make_gap_entry(
    text: str = "Users need a way to export their data in CSV format",
    classification: str = "new",
    confidence: float = 0.9,
    item_type: str = "feature_request",
) -> GapReportEntry:
    item = make_item(text=text, item_type=item_type, confidence=confidence)
    return GapReportEntry(
        item=item,
        classification=classification,
        confidence=confidence,
    )


# --- Tests ---


class TestStoryWriterAgentBasicGeneration:
    """Test basic user story generation from gap report entries."""

    @pytest.mark.asyncio
    async def test_generates_story_from_new_item(self) -> None:
        agent = StoryWriterAgent(llm_tool=MockLLMGenerationTool())
        entries = [make_gap_entry()]
        stories = await agent.generate_stories(entries)

        assert len(stories) == 1
        story = stories[0]
        assert story.title == "User Authentication"
        assert "As a" in story.user_story
        assert "I want" in story.user_story
        assert "so that" in story.user_story
        assert 2 <= len(story.acceptance_criteria) <= 10
        assert 1 <= len(story.tags) <= 5

    @pytest.mark.asyncio
    async def test_generates_stories_from_conflict_items(self) -> None:
        agent = StoryWriterAgent(llm_tool=MockLLMGenerationTool())
        entries = [make_gap_entry(classification="conflict")]
        stories = await agent.generate_stories(entries)

        assert len(stories) == 1

    @pytest.mark.asyncio
    async def test_filters_out_duplicate_items(self) -> None:
        agent = StoryWriterAgent(llm_tool=MockLLMGenerationTool())
        entries = [
            make_gap_entry(classification="duplicate"),
            make_gap_entry(classification="new", text="New feature"),
        ]
        stories = await agent.generate_stories(entries)

        assert len(stories) == 1

    @pytest.mark.asyncio
    async def test_filters_out_unprocessed_items(self) -> None:
        agent = StoryWriterAgent(llm_tool=MockLLMGenerationTool())
        entries = [make_gap_entry(classification="unprocessed")]
        stories = await agent.generate_stories(entries)

        assert len(stories) == 0

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self) -> None:
        agent = StoryWriterAgent(llm_tool=MockLLMGenerationTool())
        stories = await agent.generate_stories([])

        assert stories == []


class TestStoryWriterAgentInsufficientDetail:
    """Test handling of items with insufficient detail (Requirement 5.6)."""

    @pytest.mark.asyncio
    async def test_short_text_produces_placeholder_story(self) -> None:
        agent = StoryWriterAgent(llm_tool=MockLLMGenerationTool())
        entries = [make_gap_entry(text="Fix bug")]  # Less than 10 chars
        stories = await agent.generate_stories(entries)

        assert len(stories) == 1
        story = stories[0]
        assert story.needs_refinement is True
        assert "needs-refinement" in story.tags
        assert "[placeholder]" in story.user_story

    @pytest.mark.asyncio
    async def test_low_confidence_produces_placeholder_story(self) -> None:
        agent = StoryWriterAgent(llm_tool=MockLLMGenerationTool())
        entries = [make_gap_entry(text="Some vague requirement", confidence=0.2)]
        # Create with low confidence on the item
        entries[0].item = make_item(text="Some vague requirement", confidence=0.2)
        stories = await agent.generate_stories(entries)

        assert len(stories) == 1
        story = stories[0]
        assert story.needs_refinement is True
        assert "needs-refinement" in story.tags


class TestStoryWriterAgentLLMFailure:
    """Test graceful handling of LLM failures."""

    @pytest.mark.asyncio
    async def test_llm_failure_produces_placeholder_story(self) -> None:
        agent = StoryWriterAgent(llm_tool=FailingLLMTool())
        entries = [make_gap_entry()]
        stories = await agent.generate_stories(entries)

        assert len(stories) == 1
        story = stories[0]
        assert story.needs_refinement is True
        assert "needs-refinement" in story.tags


class TestStoryWriterAgentMultipleItems:
    """Test generating multiple stories from multiple items."""

    @pytest.mark.asyncio
    async def test_generates_multiple_stories(self) -> None:
        responses = [
            json.dumps({
                "title": "Data Export",
                "user_story": "As a user, I want to export data, so that I can analyze it offline",
                "acceptance_criteria": [
                    {"description": "Export button is available on the dashboard"},
                    {"description": "Data is exported in CSV format"},
                ],
                "tags": ["data-export", "csv"],
            }),
            json.dumps({
                "title": "Notifications",
                "user_story": "As a user, I want to receive notifications, so that I stay informed",
                "acceptance_criteria": [
                    {"description": "User receives email notifications"},
                    {"description": "User can configure notification preferences"},
                    {"description": "Notifications are sent within 5 minutes"},
                ],
                "tags": ["notifications", "email"],
            }),
        ]
        agent = StoryWriterAgent(llm_tool=StatefulLLMTool(responses))
        entries = [
            make_gap_entry(text="Users need CSV data export functionality"),
            make_gap_entry(text="System should send email notifications to users"),
        ]
        stories = await agent.generate_stories(entries)

        assert len(stories) == 2
        assert stories[0].title == "Data Export"
        assert stories[1].title == "Notifications"


class TestParseLLMResponse:
    """Test the _parse_llm_response helper function."""

    def test_parses_valid_json_response(self) -> None:
        response = json.dumps({
            "title": "Feature X",
            "user_story": "As a dev, I want feature X, so that I can do Y",
            "acceptance_criteria": [
                {"description": "Criterion 1"},
                {"description": "Criterion 2"},
            ],
            "tags": ["feature-x"],
        })
        item = make_item()
        story = _parse_llm_response(response, item)

        assert story.title == "Feature X"
        assert story.user_story == "As a dev, I want feature X, so that I can do Y"
        assert len(story.acceptance_criteria) == 2
        assert story.tags == ["feature-x"]
        assert story.needs_refinement is False

    def test_handles_markdown_code_fences(self) -> None:
        inner_json = json.dumps({
            "title": "Feature Y",
            "user_story": "As a user, I want feature Y, so that I benefit",
            "acceptance_criteria": [
                {"description": "AC 1"},
                {"description": "AC 2"},
            ],
            "tags": ["feature-y"],
        })
        response = f"```json\n{inner_json}\n```"
        item = make_item()
        story = _parse_llm_response(response, item)

        assert story.title == "Feature Y"

    def test_invalid_json_produces_placeholder(self) -> None:
        response = "This is not valid JSON at all"
        item = make_item()
        story = _parse_llm_response(response, item)

        assert story.needs_refinement is True
        assert "needs-refinement" in story.tags

    def test_fewer_than_2_criteria_gets_padded(self) -> None:
        response = json.dumps({
            "title": "Minimal",
            "user_story": "As a user, I want something, so that I benefit",
            "acceptance_criteria": [{"description": "Only one"}],
            "tags": ["minimal"],
        })
        item = make_item()
        story = _parse_llm_response(response, item)

        assert len(story.acceptance_criteria) >= 2

    def test_more_than_10_criteria_gets_capped(self) -> None:
        criteria = [{"description": f"Criterion {i}"} for i in range(15)]
        response = json.dumps({
            "title": "Many Criteria",
            "user_story": "As a user, I want many things, so that I am happy",
            "acceptance_criteria": criteria,
            "tags": ["many"],
        })
        item = make_item()
        story = _parse_llm_response(response, item)

        assert len(story.acceptance_criteria) <= 10

    def test_empty_tags_gets_default_tag(self) -> None:
        response = json.dumps({
            "title": "No Tags",
            "user_story": "As a user, I want something, so that I benefit",
            "acceptance_criteria": [
                {"description": "AC 1"},
                {"description": "AC 2"},
            ],
            "tags": [],
        })
        item = make_item(item_type="feature_request")
        story = _parse_llm_response(response, item)

        assert len(story.tags) >= 1
        assert "feature-request" in story.tags

    def test_more_than_5_tags_gets_capped(self) -> None:
        response = json.dumps({
            "title": "Many Tags",
            "user_story": "As a user, I want something, so that I benefit",
            "acceptance_criteria": [
                {"description": "AC 1"},
                {"description": "AC 2"},
            ],
            "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"],
        })
        item = make_item()
        story = _parse_llm_response(response, item)

        assert len(story.tags) <= 5

    def test_placeholder_in_story_sets_needs_refinement(self) -> None:
        response = json.dumps({
            "title": "Vague Story",
            "user_story": "As a [placeholder], I want [placeholder], so that [placeholder]",
            "acceptance_criteria": [
                {"description": "AC 1"},
                {"description": "AC 2"},
            ],
            "tags": ["vague"],
        })
        item = make_item()
        story = _parse_llm_response(response, item)

        assert story.needs_refinement is True
        assert "needs-refinement" in story.tags


class TestHasInsufficientDetail:
    """Test the _has_insufficient_detail helper function."""

    def test_short_text_is_insufficient(self) -> None:
        item = make_item(text="Fix it")
        assert _has_insufficient_detail(item) is True

    def test_low_confidence_is_insufficient(self) -> None:
        item = make_item(text="Some longer text that is detailed enough", confidence=0.2)
        assert _has_insufficient_detail(item) is True

    def test_adequate_text_is_sufficient(self) -> None:
        item = make_item(text="Users need the ability to export reports as PDF", confidence=0.8)
        assert _has_insufficient_detail(item) is False


class TestCreatePlaceholderStory:
    """Test the _create_placeholder_story helper function."""

    def test_placeholder_has_needs_refinement_tag(self) -> None:
        item = make_item(text="short")
        story = _create_placeholder_story(item)

        assert "needs-refinement" in story.tags
        assert story.needs_refinement is True

    def test_placeholder_has_valid_structure(self) -> None:
        item = make_item(text="short")
        story = _create_placeholder_story(item)

        assert "As a" in story.user_story
        assert "I want" in story.user_story
        assert "so that" in story.user_story
        assert 2 <= len(story.acceptance_criteria) <= 10
        assert 1 <= len(story.tags) <= 5
