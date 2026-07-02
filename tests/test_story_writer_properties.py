"""Property-based tests for story writer epic grouping logic.

Uses Hypothesis to verify that stories with shared tags are grouped
under a common Epic with a title of at most 60 characters.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from backlog_synthesizer.agents.story_writer import group_stories_into_epics
from backlog_synthesizer.models.output import AcceptanceCriterion, UserStory

# --- Strategies ---

# Controlled tag alphabet to ensure overlap is likely
TAG_ALPHABET = ["auth", "ui", "api", "db", "perf", "security", "logging", "cache"]

tag_strategy = st.sampled_from(TAG_ALPHABET)

# Strategy for lists of tags (1-5 tags per story, drawn from controlled alphabet)
tags_strategy = st.lists(tag_strategy, min_size=1, max_size=5, unique=True)

# Strategy for acceptance criteria (2-10 per story)
acceptance_criterion_strategy = st.builds(
    AcceptanceCriterion,
    description=st.text(min_size=1, max_size=100).filter(lambda t: t.strip() != ""),
)

acceptance_criteria_strategy = st.lists(
    acceptance_criterion_strategy, min_size=2, max_size=10
)

# Strategy for valid UserStory instances
user_story_strategy = st.builds(
    UserStory,
    title=st.text(min_size=1, max_size=80).filter(lambda t: t.strip() != ""),
    user_story=st.text(min_size=1, max_size=200).filter(lambda t: t.strip() != ""),
    acceptance_criteria=acceptance_criteria_strategy,
    tags=tags_strategy,
    needs_refinement=st.booleans(),
)

# Strategy for lists of stories (at least 2 to test grouping)
stories_strategy = st.lists(user_story_strategy, min_size=2, max_size=15)


# --- Property Tests ---


# Feature: backlog-synthesizer, Property 8: Epic grouping by shared tags
class TestProperty8EpicGroupingBySharedTags:
    """Verify stories with shared tags appear under a common Epic with title ≤ 60 chars.

    For any set of generated UserStories where two or more stories share at least
    one tag, those stories SHALL appear under a common Epic whose `epic_title` is
    at most 60 characters long.

    **Validates: Requirements 5.4**
    """

    @given(stories=stories_strategy)
    @settings(max_examples=100)
    def test_stories_sharing_tag_are_in_same_epic(
        self, stories: list[UserStory]
    ) -> None:
        """
        Stories that share at least one tag SHALL appear under a common Epic.

        **Validates: Requirements 5.4**
        """
        epics = group_stories_into_epics(stories)

        # Build a mapping from story title to its epic index
        story_to_epic: dict[str, int] = {}
        for epic_idx, epic in enumerate(epics):
            for story in epic.stories:
                story_to_epic[id(story)] = epic_idx

        # For each pair of stories that share a tag, verify same epic
        for i, story_a in enumerate(stories):
            for j, story_b in enumerate(stories):
                if i >= j:
                    continue
                shared_tags = set(story_a.tags) & set(story_b.tags)
                if shared_tags:
                    assert story_to_epic[id(story_a)] == story_to_epic[id(story_b)], (
                        f"Stories '{story_a.title}' and '{story_b.title}' share tags "
                        f"{shared_tags} but are in different epics"
                    )

    @given(stories=stories_strategy)
    @settings(max_examples=100)
    def test_all_epic_titles_at_most_60_chars(
        self, stories: list[UserStory]
    ) -> None:
        """
        All Epic titles SHALL be at most 60 characters long.

        **Validates: Requirements 5.4**
        """
        epics = group_stories_into_epics(stories)

        for epic in epics:
            assert len(epic.epic_title) <= 60, (
                f"Epic title '{epic.epic_title}' has {len(epic.epic_title)} chars, "
                f"exceeds 60 char limit"
            )

    @given(stories=stories_strategy)
    @settings(max_examples=100)
    def test_transitive_grouping(self, stories: list[UserStory]) -> None:
        """
        Grouping SHALL be transitive: if A shares a tag with B, and B shares a
        tag with C, then A, B, and C are in the same epic.

        **Validates: Requirements 5.4**
        """
        epics = group_stories_into_epics(stories)

        # Build a mapping from story identity to its epic index
        story_to_epic: dict[int, int] = {}
        for epic_idx, epic in enumerate(epics):
            for story in epic.stories:
                story_to_epic[id(story)] = epic_idx

        # Build transitive closure of "shares a tag" relationship
        n = len(stories)
        # Use union-find to compute expected groups
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        # Build tag -> indices mapping
        tag_to_indices: dict[str, list[int]] = {}
        for i, story in enumerate(stories):
            for tag in story.tags:
                if tag not in tag_to_indices:
                    tag_to_indices[tag] = []
                tag_to_indices[tag].append(i)

        # Union stories sharing tags
        for indices in tag_to_indices.values():
            for k in range(1, len(indices)):
                union(indices[0], indices[k])

        # Verify: stories in the same expected group are in the same epic
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) == find(j):
                    assert story_to_epic[id(stories[i])] == story_to_epic[id(stories[j])], (
                        f"Stories at index {i} and {j} should be transitively grouped "
                        f"but are in different epics"
                    )

    @given(stories=stories_strategy)
    @settings(max_examples=100)
    def test_all_stories_appear_exactly_once(
        self, stories: list[UserStory]
    ) -> None:
        """
        All input stories SHALL appear in exactly one epic (no stories lost,
        no duplicates).

        **Validates: Requirements 5.4**
        """
        epics = group_stories_into_epics(stories)

        # Collect all stories from epics
        all_epic_stories: list[UserStory] = []
        for epic in epics:
            all_epic_stories.extend(epic.stories)

        # Same count as input
        assert len(all_epic_stories) == len(stories), (
            f"Expected {len(stories)} stories in epics, got {len(all_epic_stories)}"
        )

        # Each input story appears exactly once (by identity)
        input_ids = {id(s) for s in stories}
        output_ids = {id(s) for s in all_epic_stories}
        assert input_ids == output_ids, "Stories were lost or duplicated during grouping"
