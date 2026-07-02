"""Property-based tests for gap detection classification logic.

Uses Hypothesis to verify that gap classification correctly applies
similarity thresholds and handles the empty backlog case.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from backlog_synthesizer.agents.gap_detection import (
    CONFLICT_LOWER_THRESHOLD,
    DUPLICATE_THRESHOLD,
    classify_item,
    classify_items_empty_backlog,
)
from backlog_synthesizer.models.extraction import ExtractedItem

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

# Strategy for ticket IDs
ticket_id_strategy = st.text(min_size=1, max_size=50).filter(lambda t: t.strip() != "")

# Similarity score strategies for specific ranges
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

# Strategy for any valid similarity score
any_score_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for contradiction descriptions
contradiction_description_strategy = st.text(min_size=1, max_size=200).filter(
    lambda t: t.strip() != ""
)


# --- Property Tests ---


# Feature: backlog-synthesizer, Property 5: Gap classification correctness by similarity threshold
class TestProperty5GapClassificationByThreshold:
    """Verify classification logic correctly applies similarity thresholds.

    For any extracted item compared against existing backlog tickets:
    (a) if the highest semantic similarity score is >= 0.85, it SHALL be classified
        as "duplicate" with the matching ticket ID,
    (b) if the score is in [0.50, 0.85) and contradicting statements are present,
        it SHALL be classified as "conflict" with a contradiction description,
    (c) otherwise it SHALL be classified as "new".
    All classifications SHALL have a confidence score in [0.0, 1.0].

    **Validates: Requirements 4.2, 4.3, 4.4**
    """

    @given(
        item=extracted_item_strategy,
        score=duplicate_score_strategy,
        ticket_id=ticket_id_strategy,
    )
    @settings(max_examples=100)
    def test_high_similarity_classified_as_duplicate(
        self, item: ExtractedItem, score: float, ticket_id: str
    ) -> None:
        """
        When similarity score >= 0.85, the item SHALL be classified as "duplicate"
        with the matching ticket ID in duplicate_info.

        **Validates: Requirements 4.2**
        """
        entry = classify_item(
            item=item,
            similarity_score=score,
            matching_ticket_id=ticket_id,
            has_contradiction=False,
        )
        assert entry.classification == "duplicate"
        assert entry.duplicate_info is not None
        assert entry.duplicate_info.matching_ticket_id == ticket_id
        assert entry.duplicate_info.similarity_score == score

    @given(
        item=extracted_item_strategy,
        score=conflict_score_strategy,
        ticket_id=ticket_id_strategy,
        contradiction_desc=contradiction_description_strategy,
    )
    @settings(max_examples=100)
    def test_mid_similarity_with_contradiction_classified_as_conflict(
        self,
        item: ExtractedItem,
        score: float,
        ticket_id: str,
        contradiction_desc: str,
    ) -> None:
        """
        When similarity score is in [0.50, 0.85) and contradicting statements
        are present, the item SHALL be classified as "conflict" with conflict_info
        containing a contradiction description.

        **Validates: Requirements 4.3**
        """
        entry = classify_item(
            item=item,
            similarity_score=score,
            matching_ticket_id=ticket_id,
            has_contradiction=True,
            contradiction_description=contradiction_desc,
        )
        assert entry.classification == "conflict"
        assert entry.conflict_info is not None
        assert entry.conflict_info.item_b_ticket_id == ticket_id
        assert entry.conflict_info.similarity_score == score
        assert len(entry.conflict_info.contradiction_description) > 0

    @given(
        item=extracted_item_strategy,
        score=new_item_score_strategy,
        ticket_id=ticket_id_strategy,
    )
    @settings(max_examples=100)
    def test_low_similarity_classified_as_new(
        self, item: ExtractedItem, score: float, ticket_id: str
    ) -> None:
        """
        When similarity score < 0.50, the item SHALL be classified as "new"
        regardless of contradiction presence.

        **Validates: Requirements 4.4**
        """
        entry = classify_item(
            item=item,
            similarity_score=score,
            matching_ticket_id=ticket_id,
            has_contradiction=True,
        )
        assert entry.classification == "new"
        assert entry.duplicate_info is None
        assert entry.conflict_info is None

    @given(
        item=extracted_item_strategy,
        score=conflict_score_strategy,
        ticket_id=ticket_id_strategy,
    )
    @settings(max_examples=100)
    def test_mid_similarity_without_contradiction_classified_as_new(
        self, item: ExtractedItem, score: float, ticket_id: str
    ) -> None:
        """
        When similarity score is in [0.50, 0.85) but no contradictions are present,
        the item SHALL be classified as "new".

        **Validates: Requirements 4.4**
        """
        entry = classify_item(
            item=item,
            similarity_score=score,
            matching_ticket_id=ticket_id,
            has_contradiction=False,
        )
        assert entry.classification == "new"
        assert entry.conflict_info is None

    @given(
        item=extracted_item_strategy,
        score=any_score_strategy,
        ticket_id=st.one_of(st.none(), ticket_id_strategy),
        has_contradiction=st.booleans(),
    )
    @settings(max_examples=100)
    def test_all_classifications_have_valid_confidence(
        self,
        item: ExtractedItem,
        score: float,
        ticket_id: str | None,
        has_contradiction: bool,
    ) -> None:
        """
        All classifications SHALL have a confidence score in [0.0, 1.0].

        **Validates: Requirements 4.2, 4.3, 4.4**
        """
        entry = classify_item(
            item=item,
            similarity_score=score,
            matching_ticket_id=ticket_id,
            has_contradiction=has_contradiction,
        )
        assert 0.0 <= entry.confidence <= 1.0


# Feature: backlog-synthesizer, Property 6: Empty backlog marks all items as new
class TestProperty6EmptyBacklogAllNew:
    """Verify that with an empty backlog, all items are classified as "new" with confidence 1.0.

    For any list of extracted items and an empty set of existing backlog tickets,
    the Gap_Detection_Agent SHALL classify every item as "new" with a confidence
    score of exactly 1.0.

    **Validates: Requirements 4.5**
    """

    @given(items=st.lists(extracted_item_strategy, min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_all_items_classified_as_new(self, items: list[ExtractedItem]) -> None:
        """
        For any list of extracted items with an empty backlog, every item
        SHALL be classified as "new".

        **Validates: Requirements 4.5**
        """
        report = classify_items_empty_backlog(items)
        for entry in report.entries:
            assert entry.classification == "new"

    @given(items=st.lists(extracted_item_strategy, min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_all_items_have_confidence_one(self, items: list[ExtractedItem]) -> None:
        """
        For any list of extracted items with an empty backlog, every item
        SHALL have confidence exactly 1.0.

        **Validates: Requirements 4.5**
        """
        report = classify_items_empty_backlog(items)
        for entry in report.entries:
            assert entry.confidence == 1.0

    @given(items=st.lists(extracted_item_strategy, min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_report_counts_match_all_new(self, items: list[ExtractedItem]) -> None:
        """
        For any list of extracted items with an empty backlog, the gap report
        SHALL have total_new equal to the number of items and zero duplicates,
        conflicts, and unprocessed.

        **Validates: Requirements 4.5**
        """
        report = classify_items_empty_backlog(items)
        assert report.total_new == len(items)
        assert report.total_duplicates == 0
        assert report.total_conflicts == 0
        assert report.total_unprocessed == 0
        assert len(report.entries) == len(items)

    @given(items=st.lists(extracted_item_strategy, min_size=0, max_size=0))
    @settings(max_examples=100)
    def test_empty_items_produces_empty_report(self, items: list[ExtractedItem]) -> None:
        """
        For an empty list of items with an empty backlog, the gap report
        SHALL have zero entries and all counts at zero.

        **Validates: Requirements 4.5**
        """
        report = classify_items_empty_backlog(items)
        assert len(report.entries) == 0
        assert report.total_new == 0
        assert report.total_duplicates == 0
        assert report.total_conflicts == 0
        assert report.total_unprocessed == 0
