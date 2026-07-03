"""Tests for gap classification output fields: gap_type, similarity_score, similar_ticket_id.

Verifies that the enhanced GapReportEntry model correctly populates all
classification-related fields for DUPLICATE, CONFLICT, and NEW items.

Validates: Requirements 4.2, 4.3, 4.4, 4.5
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backlog_synthesizer.agents.gap_detection import (
    CONFLICT_LOWER_THRESHOLD,
    DUPLICATE_THRESHOLD,
    classify_item,
    classify_items_empty_backlog,
)
from backlog_synthesizer.models.extraction import ExtractedItem
from backlog_synthesizer.models.gap_detection import GapReportEntry


# --- Strategies ---

item_types = st.sampled_from(["decision", "pain_point", "feature_request", "constraint"])
non_empty_text = st.text(min_size=1, max_size=200).filter(lambda t: t.strip() != "")

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

ticket_id_strategy = st.text(min_size=1, max_size=50).filter(lambda t: t.strip() != "")

duplicate_score_strategy = st.floats(
    min_value=DUPLICATE_THRESHOLD, max_value=1.0, allow_nan=False, allow_infinity=False
)

conflict_score_strategy = st.floats(
    min_value=CONFLICT_LOWER_THRESHOLD,
    max_value=DUPLICATE_THRESHOLD,
    allow_nan=False,
    allow_infinity=False,
    exclude_max=True,
)

new_item_score_strategy = st.floats(
    min_value=0.0,
    max_value=CONFLICT_LOWER_THRESHOLD,
    allow_nan=False,
    allow_infinity=False,
    exclude_max=True,
)


# --- Tests for DUPLICATE classification fields ---


class TestDuplicateClassificationFields:
    """Verify all output fields are correctly populated for DUPLICATE items."""

    @given(
        item=extracted_item_strategy,
        score=duplicate_score_strategy,
        ticket_id=ticket_id_strategy,
    )
    @settings(max_examples=100)
    def test_gap_type_is_duplicate(self, item, score, ticket_id):
        """DUPLICATE items have gap_type='DUPLICATE'."""
        entry = classify_item(item, score, ticket_id, has_contradiction=False)
        assert entry.gap_type == "DUPLICATE"

    @given(
        item=extracted_item_strategy,
        score=duplicate_score_strategy,
        ticket_id=ticket_id_strategy,
    )
    @settings(max_examples=100)
    def test_similarity_score_matches_input(self, item, score, ticket_id):
        """DUPLICATE items have similarity_score equal to the input score."""
        entry = classify_item(item, score, ticket_id, has_contradiction=False)
        assert entry.similarity_score == score

    @given(
        item=extracted_item_strategy,
        score=duplicate_score_strategy,
        ticket_id=ticket_id_strategy,
    )
    @settings(max_examples=100)
    def test_similar_ticket_id_matches(self, item, score, ticket_id):
        """DUPLICATE items have similar_ticket_id set to the matching ticket."""
        entry = classify_item(item, score, ticket_id, has_contradiction=False)
        assert entry.similar_ticket_id == ticket_id

    @given(
        item=extracted_item_strategy,
        score=duplicate_score_strategy,
        ticket_id=ticket_id_strategy,
    )
    @settings(max_examples=100)
    def test_confidence_equals_similarity(self, item, score, ticket_id):
        """DUPLICATE items have confidence equal to similarity_score."""
        entry = classify_item(item, score, ticket_id, has_contradiction=False)
        assert entry.confidence == score


# --- Tests for CONFLICT classification fields ---


class TestConflictClassificationFields:
    """Verify all output fields are correctly populated for CONFLICT items."""

    @given(
        item=extracted_item_strategy,
        score=conflict_score_strategy,
        ticket_id=ticket_id_strategy,
    )
    @settings(max_examples=100)
    def test_gap_type_is_conflict(self, item, score, ticket_id):
        """CONFLICT items have gap_type='CONFLICT'."""
        entry = classify_item(
            item, score, ticket_id,
            has_contradiction=True,
            contradiction_description="They disagree",
        )
        assert entry.gap_type == "CONFLICT"

    @given(
        item=extracted_item_strategy,
        score=conflict_score_strategy,
        ticket_id=ticket_id_strategy,
    )
    @settings(max_examples=100)
    def test_similarity_score_matches_input(self, item, score, ticket_id):
        """CONFLICT items have similarity_score equal to the input score."""
        entry = classify_item(
            item, score, ticket_id,
            has_contradiction=True,
            contradiction_description="Contradiction found",
        )
        assert entry.similarity_score == score

    @given(
        item=extracted_item_strategy,
        score=conflict_score_strategy,
        ticket_id=ticket_id_strategy,
    )
    @settings(max_examples=100)
    def test_similar_ticket_id_matches(self, item, score, ticket_id):
        """CONFLICT items have similar_ticket_id set to the conflicting ticket."""
        entry = classify_item(
            item, score, ticket_id,
            has_contradiction=True,
            contradiction_description="Contradiction found",
        )
        assert entry.similar_ticket_id == ticket_id

    @given(
        item=extracted_item_strategy,
        score=conflict_score_strategy,
        ticket_id=ticket_id_strategy,
    )
    @settings(max_examples=100)
    def test_confidence_equals_similarity(self, item, score, ticket_id):
        """CONFLICT items have confidence equal to similarity_score."""
        entry = classify_item(
            item, score, ticket_id,
            has_contradiction=True,
            contradiction_description="Contradiction found",
        )
        assert entry.confidence == score


# --- Tests for NEW classification fields ---


class TestNewClassificationFields:
    """Verify all output fields are correctly populated for NEW items."""

    @given(
        item=extracted_item_strategy,
        score=new_item_score_strategy,
        ticket_id=ticket_id_strategy,
    )
    @settings(max_examples=100)
    def test_gap_type_is_new(self, item, score, ticket_id):
        """NEW items have gap_type='NEW'."""
        entry = classify_item(item, score, ticket_id, has_contradiction=False)
        assert entry.gap_type == "NEW"

    @given(
        item=extracted_item_strategy,
        score=new_item_score_strategy,
        ticket_id=ticket_id_strategy,
    )
    @settings(max_examples=100)
    def test_similarity_score_populated(self, item, score, ticket_id):
        """NEW items have similarity_score set to the actual score."""
        entry = classify_item(item, score, ticket_id, has_contradiction=False)
        assert entry.similarity_score == score

    @given(
        item=extracted_item_strategy,
        score=new_item_score_strategy,
        ticket_id=ticket_id_strategy,
    )
    @settings(max_examples=100)
    def test_similar_ticket_id_populated(self, item, score, ticket_id):
        """NEW items still reference the closest ticket (even if below threshold)."""
        entry = classify_item(item, score, ticket_id, has_contradiction=False)
        assert entry.similar_ticket_id == ticket_id

    @given(item=extracted_item_strategy)
    @settings(max_examples=100)
    def test_no_ticket_gives_none(self, item):
        """NEW items with no matching ticket have similar_ticket_id=None."""
        entry = classify_item(item, 0.0, None, has_contradiction=False)
        assert entry.similar_ticket_id is None
        assert entry.similarity_score == 0.0
        assert entry.gap_type == "NEW"


# --- Tests for empty backlog ---


class TestEmptyBacklogFields:
    """Verify fields on items classified via the empty backlog path."""

    @given(items=st.lists(extracted_item_strategy, min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_all_entries_have_new_gap_type(self, items):
        """Empty backlog produces entries with gap_type='NEW'."""
        report = classify_items_empty_backlog(items)
        for entry in report.entries:
            assert entry.gap_type == "NEW"

    @given(items=st.lists(extracted_item_strategy, min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_all_entries_have_zero_similarity(self, items):
        """Empty backlog produces entries with similarity_score=0.0."""
        report = classify_items_empty_backlog(items)
        for entry in report.entries:
            assert entry.similarity_score == 0.0

    @given(items=st.lists(extracted_item_strategy, min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_all_entries_have_no_similar_ticket(self, items):
        """Empty backlog produces entries with similar_ticket_id=None."""
        report = classify_items_empty_backlog(items)
        for entry in report.entries:
            assert entry.similar_ticket_id is None


# --- Test configurable thresholds ---


class TestConfigurableThresholds:
    """Verify thresholds are read from environment and applied correctly."""

    def test_default_duplicate_threshold(self):
        """Default duplicate threshold is 0.80."""
        assert DUPLICATE_THRESHOLD == 0.80

    def test_default_conflict_threshold(self):
        """Default conflict lower threshold is 0.50."""
        assert CONFLICT_LOWER_THRESHOLD == 0.50

    def test_boundary_at_duplicate_threshold(self):
        """Score exactly at duplicate threshold yields DUPLICATE."""
        item = ExtractedItem(
            item_type="feature_request",
            text="Add dark mode support",
            source_chunk_index=0,
            confidence=0.9,
        )
        entry = classify_item(item, DUPLICATE_THRESHOLD, "TICKET-1", has_contradiction=False)
        assert entry.gap_type == "DUPLICATE"

    def test_boundary_just_below_duplicate_threshold_with_contradiction(self):
        """Score just below duplicate threshold with contradiction yields CONFLICT."""
        item = ExtractedItem(
            item_type="feature_request",
            text="Remove dark mode support",
            source_chunk_index=0,
            confidence=0.9,
        )
        score = DUPLICATE_THRESHOLD - 0.01
        entry = classify_item(
            item, score, "TICKET-1",
            has_contradiction=True,
            contradiction_description="Opposite intent",
        )
        assert entry.gap_type == "CONFLICT"

    def test_boundary_just_below_conflict_threshold(self):
        """Score just below conflict threshold yields NEW."""
        item = ExtractedItem(
            item_type="feature_request",
            text="Brand new feature",
            source_chunk_index=0,
            confidence=0.9,
        )
        score = CONFLICT_LOWER_THRESHOLD - 0.01
        entry = classify_item(item, score, "TICKET-1", has_contradiction=True)
        assert entry.gap_type == "NEW"
