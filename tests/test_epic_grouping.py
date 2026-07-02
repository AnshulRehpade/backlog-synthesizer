"""Unit tests for epic grouping logic.

Tests the group_stories_into_epics function and StoryWriterAgent.group_into_epics
method to verify correct tag-based grouping, transitive grouping, title generation,
and edge case handling.

Requirements: 5.4
"""

import pytest

from backlog_synthesizer.agents.story_writer import (
    StoryWriterAgent,
    _generate_epic_title,
    group_stories_into_epics,
)
from backlog_synthesizer.models.output import AcceptanceCriterion, Epic, UserStory


# --- Helpers ---


def make_story(
    title: str = "Test Story",
    tags: list[str] | None = None,
    user_story: str = "As a user, I want something, so that I benefit",
) -> UserStory:
    """Create a UserStory with sensible defaults for testing."""
    if tags is None:
        tags = ["default-tag"]
    return UserStory(
        title=title,
        user_story=user_story,
        acceptance_criteria=[
            AcceptanceCriterion(description="First criterion"),
            AcceptanceCriterion(description="Second criterion"),
        ],
        tags=tags,
        needs_refinement=False,
    )


# --- Tests ---


class TestEpicGroupingBasic:
    """Test basic epic grouping behavior."""

    def test_stories_with_shared_tag_grouped_together(self) -> None:
        """Stories sharing at least one tag should be in the same epic."""
        stories = [
            make_story(title="Story A", tags=["auth", "security"]),
            make_story(title="Story B", tags=["auth", "login"]),
        ]
        epics = group_stories_into_epics(stories)

        assert len(epics) == 1
        assert len(epics[0].stories) == 2
        assert stories[0] in epics[0].stories
        assert stories[1] in epics[0].stories

    def test_stories_with_no_shared_tags_get_separate_epics(self) -> None:
        """Stories with completely different tags get their own epics."""
        stories = [
            make_story(title="Story A", tags=["auth"]),
            make_story(title="Story B", tags=["payments"]),
            make_story(title="Story C", tags=["notifications"]),
        ]
        epics = group_stories_into_epics(stories)

        assert len(epics) == 3
        # Each epic should have exactly 1 story
        for epic in epics:
            assert len(epic.stories) == 1

    def test_empty_input_returns_empty_list(self) -> None:
        """No stories means no epics."""
        epics = group_stories_into_epics([])
        assert epics == []

    def test_single_story_gets_its_own_epic(self) -> None:
        """A single story should be placed in its own epic."""
        stories = [make_story(title="Lone Story", tags=["solo"])]
        epics = group_stories_into_epics(stories)

        assert len(epics) == 1
        assert len(epics[0].stories) == 1
        assert epics[0].stories[0].title == "Lone Story"


class TestEpicGroupingTransitive:
    """Test transitive grouping via union-find."""

    def test_transitive_grouping_a_b_c(self) -> None:
        """If A shares a tag with B, and B shares a tag with C, all three are in one epic."""
        stories = [
            make_story(title="Story A", tags=["tag-x"]),
            make_story(title="Story B", tags=["tag-x", "tag-y"]),
            make_story(title="Story C", tags=["tag-y"]),
        ]
        epics = group_stories_into_epics(stories)

        assert len(epics) == 1
        assert len(epics[0].stories) == 3

    def test_transitive_chain_of_four(self) -> None:
        """A chain A-B-C-D through shared tags should all group together."""
        stories = [
            make_story(title="Story A", tags=["t1"]),
            make_story(title="Story B", tags=["t1", "t2"]),
            make_story(title="Story C", tags=["t2", "t3"]),
            make_story(title="Story D", tags=["t3"]),
        ]
        epics = group_stories_into_epics(stories)

        assert len(epics) == 1
        assert len(epics[0].stories) == 4

    def test_two_separate_groups(self) -> None:
        """Two disconnected groups of stories produce two epics."""
        stories = [
            make_story(title="Story A", tags=["group1"]),
            make_story(title="Story B", tags=["group1"]),
            make_story(title="Story C", tags=["group2"]),
            make_story(title="Story D", tags=["group2"]),
        ]
        epics = group_stories_into_epics(stories)

        assert len(epics) == 2
        # Find which epic contains which stories
        epic_sizes = sorted(len(e.stories) for e in epics)
        assert epic_sizes == [2, 2]


class TestEpicTitleGeneration:
    """Test epic title constraints and generation."""

    def test_epic_title_max_60_characters(self) -> None:
        """All epic titles must be at most 60 characters."""
        stories = [
            make_story(title="Story A", tags=["very-long-tag-name-one", "another-very-long-tag"]),
            make_story(title="Story B", tags=["very-long-tag-name-one", "yet-another-long-tag-name"]),
        ]
        epics = group_stories_into_epics(stories)

        for epic in epics:
            assert len(epic.epic_title) <= 60

    def test_epic_title_reflects_shared_tags(self) -> None:
        """Epic title should be derived from the shared tags."""
        stories = [
            make_story(title="Story A", tags=["authentication"]),
            make_story(title="Story B", tags=["authentication"]),
        ]
        epics = group_stories_into_epics(stories)

        assert len(epics) == 1
        # Title should contain something related to "authentication"
        assert "Authentication" in epics[0].epic_title

    def test_epic_title_with_hyphenated_tags(self) -> None:
        """Hyphens in tags should be converted to spaces in the title."""
        stories = [
            make_story(title="Story A", tags=["user-auth"]),
            make_story(title="Story B", tags=["user-auth"]),
        ]
        epics = group_stories_into_epics(stories)

        assert len(epics) == 1
        assert "User Auth" in epics[0].epic_title

    def test_single_story_epic_has_valid_title(self) -> None:
        """Even a single-story epic should have a meaningful title."""
        stories = [make_story(title="Only Story", tags=["data-export"])]
        epics = group_stories_into_epics(stories)

        assert len(epics) == 1
        assert len(epics[0].epic_title) > 0
        assert len(epics[0].epic_title) <= 60

    def test_title_truncated_when_too_long(self) -> None:
        """Title with many long tags should be truncated with ellipsis."""
        # Create stories with many long tags that would exceed 60 chars
        long_tags = [
            "very-long-feature-name-authentication",
            "another-extremely-long-tag-name",
            "third-incredibly-verbose-tag",
        ]
        stories = [
            make_story(title="Story A", tags=long_tags[:3]),
            make_story(title="Story B", tags=long_tags[:3]),
        ]
        epics = group_stories_into_epics(stories)

        for epic in epics:
            assert len(epic.epic_title) <= 60


class TestEpicGroupingWithAgent:
    """Test the StoryWriterAgent.group_into_epics method."""

    def test_agent_method_delegates_to_module_function(self) -> None:
        """The agent method should produce the same result as the standalone function."""

        class DummyLLM:
            def generate(self, prompt: str, system_prompt: str | None = None) -> str:
                return ""

        agent = StoryWriterAgent(llm_tool=DummyLLM())
        stories = [
            make_story(title="Story A", tags=["shared"]),
            make_story(title="Story B", tags=["shared"]),
        ]
        epics = agent.group_into_epics(stories)

        assert len(epics) == 1
        assert len(epics[0].stories) == 2


class TestEpicGroupingEdgeCases:
    """Test edge cases for epic grouping."""

    def test_all_stories_share_one_common_tag(self) -> None:
        """If all stories share one tag, they all go into one epic."""
        stories = [
            make_story(title=f"Story {i}", tags=["common", f"unique-{i}"])
            for i in range(5)
        ]
        epics = group_stories_into_epics(stories)

        assert len(epics) == 1
        assert len(epics[0].stories) == 5

    def test_stories_with_multiple_shared_tags(self) -> None:
        """Stories sharing multiple tags are still grouped into one epic."""
        stories = [
            make_story(title="Story A", tags=["auth", "security", "login"]),
            make_story(title="Story B", tags=["auth", "security"]),
        ]
        epics = group_stories_into_epics(stories)

        assert len(epics) == 1
        assert len(epics[0].stories) == 2

    def test_epic_model_validation(self) -> None:
        """Produced epics should be valid Epic model instances."""
        stories = [
            make_story(title="Story A", tags=["feature"]),
            make_story(title="Story B", tags=["feature"]),
        ]
        epics = group_stories_into_epics(stories)

        for epic in epics:
            assert isinstance(epic, Epic)
            assert len(epic.epic_title) <= 60
            assert len(epic.stories) >= 1
