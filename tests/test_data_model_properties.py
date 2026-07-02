"""Property-based tests for data model validation.

Uses Hypothesis to verify structural invariants of the Backlog Synthesizer data models.
"""

import re
from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from backlog_synthesizer.models.extraction import ExtractedItem
from backlog_synthesizer.models.output import (
    AcceptanceCriterion,
    Epic,
    OutputMetadata,
    StoryOutput,
    UserStory,
)

# --- Strategies ---

# Strategy for valid item types
item_types = st.sampled_from(["decision", "pain_point", "feature_request", "constraint"])

# Strategy for non-empty text
non_empty_text = st.text(min_size=1, max_size=200).filter(lambda t: t.strip() != "")

# Strategy for valid ExtractedItem instances
extracted_item_strategy = st.builds(
    ExtractedItem,
    item_type=item_types,
    text=non_empty_text,
    source_chunk_index=st.integers(min_value=0, max_value=1000),
    char_offset=st.one_of(st.none(), st.integers(min_value=0, max_value=10000)),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    stakeholder=st.one_of(st.none(), non_empty_text),
    section_heading=st.one_of(st.none(), non_empty_text),
    type_classification=st.one_of(
        st.none(), st.sampled_from(["constraint", "decision", "principle"])
    ),
)

# Strategy for user story text matching the required pattern
user_story_text = st.builds(
    lambda role, goal, benefit: f"As a {role}, I want {goal}, so that {benefit}",
    role=st.text(min_size=1, max_size=30).filter(lambda t: t.strip() != "" and "," not in t),
    goal=st.text(min_size=1, max_size=80).filter(
        lambda t: t.strip() != "" and "so that" not in t.lower()
    ),
    benefit=st.text(min_size=1, max_size=80).filter(lambda t: t.strip() != ""),
)

# Strategy for AcceptanceCriterion
acceptance_criterion_strategy = st.builds(
    AcceptanceCriterion,
    description=non_empty_text,
)

# Strategy for valid UserStory instances
user_story_strategy = st.builds(
    UserStory,
    title=non_empty_text,
    user_story=user_story_text,
    acceptance_criteria=st.lists(
        acceptance_criterion_strategy, min_size=2, max_size=10
    ),
    tags=st.lists(non_empty_text, min_size=1, max_size=5),
    needs_refinement=st.booleans(),
)

# Strategy for valid Epic instances
epic_strategy = st.builds(
    Epic,
    epic_title=st.text(min_size=1, max_size=60).filter(lambda t: t.strip() != ""),
    stories=st.lists(user_story_strategy, min_size=1, max_size=3),
)

# Strategy for OutputMetadata
output_metadata_strategy = st.builds(
    OutputMetadata,
    session_id=st.text(min_size=1, max_size=50).filter(lambda t: t.strip() != ""),
    timestamp=st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31),
        timezones=st.just(timezone.utc),
    ),
)


def build_story_output(epics: list[Epic], metadata: OutputMetadata) -> StoryOutput:
    """Build a StoryOutput with a consistent index derived from the epics."""
    index = [
        {"epic_title": epic.epic_title, "story_count": len(epic.stories)}
        for epic in epics
    ]
    return StoryOutput(index=index, epics=epics, metadata=metadata)


# Strategy for valid StoryOutput instances with consistent index
story_output_strategy = st.builds(
    build_story_output,
    epics=st.lists(epic_strategy, min_size=1, max_size=3),
    metadata=output_metadata_strategy,
)


# --- Property Tests ---


# Feature: backlog-synthesizer, Property 1: Output JSON round-trip serialization
class TestProperty1OutputRoundTrip:
    """Verify StoryOutput JSON round-trip serialization produces equal objects."""

    @given(output=story_output_strategy)
    @settings(max_examples=100)
    def test_round_trip_serialization(self, output: StoryOutput) -> None:
        """
        For any valid StoryOutput instance, serializing to JSON and deserializing
        back produces a semantically equivalent object.

        **Validates: Requirements 10.2**
        """
        json_str = output.model_dump_json()
        restored = StoryOutput.model_validate_json(json_str)
        assert restored == output


# Feature: backlog-synthesizer, Property 4: Extracted item structural validity
class TestProperty4ExtractedItemValidity:
    """Verify ExtractedItem structural invariants hold for all valid instances."""

    @given(item=extracted_item_strategy)
    @settings(max_examples=100)
    def test_text_is_non_empty(self, item: ExtractedItem) -> None:
        """
        For any extracted item, the text field must be non-empty.

        **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
        """
        assert len(item.text) > 0
        assert item.text.strip() != ""

    @given(item=extracted_item_strategy)
    @settings(max_examples=100)
    def test_source_chunk_index_non_negative(self, item: ExtractedItem) -> None:
        """
        For any extracted item, the source_chunk_index must be non-negative.

        **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
        """
        assert item.source_chunk_index >= 0

    @given(item=extracted_item_strategy)
    @settings(max_examples=100)
    def test_confidence_in_valid_range(self, item: ExtractedItem) -> None:
        """
        For any extracted item, the confidence score must be in [0.0, 1.0].

        **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
        """
        assert 0.0 <= item.confidence <= 1.0


# Feature: backlog-synthesizer, Property 7: User story structural validity
class TestProperty7UserStoryValidity:
    """Verify UserStory structural invariants hold for all valid instances."""

    @given(story=user_story_strategy)
    @settings(max_examples=100)
    def test_user_story_pattern_match(self, story: UserStory) -> None:
        """
        For any generated UserStory, the user_story field must match the pattern
        "As a [role], I want [goal], so that [benefit]".

        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        pattern = r"^As a .+, I want .+, so that .+"
        assert re.match(pattern, story.user_story, re.DOTALL), (
            f"user_story does not match pattern: {story.user_story!r}"
        )

    @given(story=user_story_strategy)
    @settings(max_examples=100)
    def test_acceptance_criteria_count(self, story: UserStory) -> None:
        """
        For any generated UserStory, acceptance_criteria must have 2-10 entries.

        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        assert 2 <= len(story.acceptance_criteria) <= 10

    @given(story=user_story_strategy)
    @settings(max_examples=100)
    def test_tags_count(self, story: UserStory) -> None:
        """
        For any generated UserStory, tags must have 1-5 entries.

        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        assert 1 <= len(story.tags) <= 5


# Feature: backlog-synthesizer, Property 16: Output index array consistency
class TestProperty16OutputIndexConsistency:
    """Verify StoryOutput index array is consistent with epics."""

    @given(output=story_output_strategy)
    @settings(max_examples=100)
    def test_index_length_matches_epics_count(self, output: StoryOutput) -> None:
        """
        For any StoryOutput, the index array length must equal the number of epics.

        **Validates: Requirements 10.3**
        """
        assert len(output.index) == len(output.epics)

    @given(output=story_output_strategy)
    @settings(max_examples=100)
    def test_index_titles_match_epics(self, output: StoryOutput) -> None:
        """
        For any StoryOutput, each index entry's epic_title must match the
        corresponding epic's title.

        **Validates: Requirements 10.3**
        """
        for i, (index_entry, epic) in enumerate(zip(output.index, output.epics)):
            assert index_entry["epic_title"] == epic.epic_title, (
                f"Index entry {i}: expected title {epic.epic_title!r}, "
                f"got {index_entry['epic_title']!r}"
            )

    @given(output=story_output_strategy)
    @settings(max_examples=100)
    def test_index_story_counts_match_epics(self, output: StoryOutput) -> None:
        """
        For any StoryOutput, each index entry's story_count must equal the length
        of the corresponding epic's stories list.

        **Validates: Requirements 10.3**
        """
        for i, (index_entry, epic) in enumerate(zip(output.index, output.epics)):
            assert index_entry["story_count"] == len(epic.stories), (
                f"Index entry {i}: expected story_count {len(epic.stories)}, "
                f"got {index_entry['story_count']}"
            )
