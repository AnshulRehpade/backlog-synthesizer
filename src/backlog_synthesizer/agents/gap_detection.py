"""Gap Detection Agent for the Backlog Synthesizer system.

Compares extracted items against existing backlog using semantic similarity
and classifies them as new, duplicate, or conflict based on thresholds.
"""

import asyncio
import os

from backlog_synthesizer.models.extraction import ExtractedItem
from backlog_synthesizer.models.gap_detection import (
    ConflictFlag,
    DuplicateFlag,
    GapReport,
    GapReportEntry,
)
from backlog_synthesizer.tools.interfaces import (
    EmbeddingTool,
    LLMGenerationTool,
    SearchResult,
    VectorSearchTool,
)


# Threshold constants — configurable via environment variables
DUPLICATE_THRESHOLD = float(os.environ.get("GAP_DETECTION_DUPLICATE_THRESHOLD", "0.85"))
CONFLICT_LOWER_THRESHOLD = float(os.environ.get("GAP_DETECTION_CONFLICT_THRESHOLD", "0.50"))


def classify_item(
    item: ExtractedItem,
    similarity_score: float,
    matching_ticket_id: str | None,
    has_contradiction: bool,
    contradiction_description: str | None = None,
) -> GapReportEntry:
    """Classify an extracted item based on similarity score and contradiction presence.

    Thresholds:
        - similarity_score >= 0.85 → "duplicate" with matching ticket ID
        - similarity_score in [0.50, 0.85) with contradictions → "conflict"
        - similarity_score < 0.50 or no contradictions → "new"

    Args:
        item: The extracted item to classify.
        similarity_score: Highest semantic similarity score against backlog (0.0 to 1.0).
        matching_ticket_id: The ID of the most similar backlog ticket, or None.
        has_contradiction: Whether contradicting statements were detected.
        contradiction_description: Description of the contradiction (if any).

    Returns:
        A GapReportEntry with the classification result.
    """
    # Clamp similarity score to valid range
    similarity_score = max(0.0, min(1.0, similarity_score))

    if similarity_score >= DUPLICATE_THRESHOLD and matching_ticket_id is not None:
        # Duplicate classification
        duplicate_info = DuplicateFlag(
            extracted_item=item,
            matching_ticket_id=matching_ticket_id,
            similarity_score=similarity_score,
        )
        return GapReportEntry(
            item=item,
            classification="duplicate",
            gap_type="DUPLICATE",
            confidence=similarity_score,
            similarity_score=similarity_score,
            similar_ticket_id=matching_ticket_id,
            duplicate_info=duplicate_info,
        )
    elif (
        CONFLICT_LOWER_THRESHOLD <= similarity_score < DUPLICATE_THRESHOLD
        and has_contradiction
        and matching_ticket_id is not None
    ):
        # Conflict classification
        conflict_info = ConflictFlag(
            item_a=item,
            item_b_ticket_id=matching_ticket_id,
            similarity_score=similarity_score,
            contradiction_description=contradiction_description or "Contradicting statements detected",
        )
        return GapReportEntry(
            item=item,
            classification="conflict",
            gap_type="CONFLICT",
            confidence=similarity_score,
            similarity_score=similarity_score,
            similar_ticket_id=matching_ticket_id,
            conflict_info=conflict_info,
        )
    else:
        # New classification
        return GapReportEntry(
            item=item,
            classification="new",
            gap_type="NEW",
            confidence=1.0 - similarity_score if matching_ticket_id is not None else 1.0,
            similarity_score=similarity_score if matching_ticket_id is not None else 0.0,
            similar_ticket_id=matching_ticket_id,
        )


def classify_items_empty_backlog(items: list[ExtractedItem]) -> GapReport:
    """Classify all items as new when no backlog tickets exist.

    When no existing backlog tickets are available for comparison,
    all items are marked as new with confidence 1.0.

    Args:
        items: List of extracted items to classify.

    Returns:
        A GapReport with all items classified as "new" with confidence 1.0.
    """
    entries = [
        GapReportEntry(
            item=item,
            classification="new",
            gap_type="NEW",
            confidence=1.0,
            similarity_score=0.0,
            similar_ticket_id=None,
        )
        for item in items
    ]
    return GapReport(
        entries=entries,
        total_new=len(items),
        total_duplicates=0,
        total_conflicts=0,
        total_unprocessed=0,
    )


class GapDetectionAgent:
    """Agent that compares extracted items against existing backlog using semantic similarity.

    Uses EmbeddingTool for vector generation, VectorSearchTool for similarity queries,
    and optionally LLMGenerationTool for contradiction detection in the conflict range.

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
    """

    def __init__(
        self,
        embedding_tool: EmbeddingTool,
        vector_search_tool: VectorSearchTool,
        llm_tool: LLMGenerationTool | None = None,
    ) -> None:
        """Initialize GapDetectionAgent with tool dependencies.

        Args:
            embedding_tool: Tool for generating text embeddings.
            vector_search_tool: Tool for querying similar items in the vector store.
            llm_tool: Optional tool for LLM-based contradiction detection.
        """
        self._embedding_tool = embedding_tool
        self._vector_search_tool = vector_search_tool
        self._llm_tool = llm_tool

    async def analyze_gaps(
        self, items: list[ExtractedItem], timeout: float = 30.0
    ) -> GapReport:
        """Compare extracted items against backlog and classify each as new, duplicate, or conflict.

        For each item:
        1. Generate embedding via EmbeddingTool.
        2. Query similar items via VectorSearchTool (top_k=5).
        3. Apply classification thresholds.
        4. Enforce per-request timeout; mark as unprocessed on failure.

        Args:
            items: List of extracted items to analyze.
            timeout: Maximum seconds allowed per item processing (default 30s).

        Returns:
            A GapReport with classified entries and summary counts.
        """
        entries: list[GapReportEntry] = []

        for item in items:
            try:
                entry = await asyncio.wait_for(
                    self._process_item(item),
                    timeout=timeout,
                )
                entries.append(entry)
            except asyncio.TimeoutError:
                entries.append(
                    GapReportEntry(
                        item=item,
                        classification="unprocessed",
                        gap_type="UNPROCESSED",
                        confidence=0.0,
                        similarity_score=0.0,
                        error_reason="Processing timed out after {:.0f} seconds".format(timeout),
                    )
                )
            except Exception as e:
                entries.append(
                    GapReportEntry(
                        item=item,
                        classification="unprocessed",
                        gap_type="UNPROCESSED",
                        confidence=0.0,
                        similarity_score=0.0,
                        error_reason=str(e),
                    )
                )

        total_new = sum(1 for e in entries if e.classification == "new")
        total_duplicates = sum(1 for e in entries if e.classification == "duplicate")
        total_conflicts = sum(1 for e in entries if e.classification == "conflict")
        total_unprocessed = sum(1 for e in entries if e.classification == "unprocessed")

        return GapReport(
            entries=entries,
            total_new=total_new,
            total_duplicates=total_duplicates,
            total_conflicts=total_conflicts,
            total_unprocessed=total_unprocessed,
        )

    async def _process_item(self, item: ExtractedItem) -> GapReportEntry:
        """Process a single item: embed, search, classify.

        Args:
            item: The extracted item to process.

        Returns:
            A classified GapReportEntry.
        """
        # Generate embedding for the item text
        embedding = await asyncio.to_thread(
            self._embedding_tool.generate_embedding, item.text
        )

        # Query vector store for similar backlog items
        results: list[SearchResult] = await asyncio.to_thread(
            self._vector_search_tool.query_similar, embedding, 5
        )

        # If no results found, classify as new (empty backlog case)
        if not results:
            return GapReportEntry(
                item=item,
                classification="new",
                gap_type="NEW",
                confidence=1.0,
                similarity_score=0.0,
                similar_ticket_id=None,
            )

        # Use the highest similarity score from results
        best_result = max(results, key=lambda r: r.score)
        similarity_score = best_result.score
        matching_ticket_id = best_result.item_id

        # Determine if contradiction exists for the conflict range
        has_contradiction = False
        contradiction_description: str | None = None

        if CONFLICT_LOWER_THRESHOLD <= similarity_score < DUPLICATE_THRESHOLD:
            # Use LLM tool to detect contradictions if available
            if self._llm_tool is not None:
                has_contradiction, contradiction_description = (
                    await self._detect_contradiction(item, best_result)
                )

        # Delegate to the existing classify_item function
        return classify_item(
            item=item,
            similarity_score=similarity_score,
            matching_ticket_id=matching_ticket_id,
            has_contradiction=has_contradiction,
            contradiction_description=contradiction_description,
        )

    async def _detect_contradiction(
        self, item: ExtractedItem, search_result: SearchResult
    ) -> tuple[bool, str | None]:
        """Use LLM to detect contradictions between item and a matching backlog ticket.

        Args:
            item: The newly extracted item.
            search_result: The search result with potential conflict.

        Returns:
            Tuple of (has_contradiction, contradiction_description).
        """
        if self._llm_tool is None:
            return False, None

        existing_text = search_result.metadata.get("text", search_result.item_id)

        prompt = (
            "Compare the following two statements and determine if they contain "
            "mutually exclusive or contradicting claims about the same feature or topic.\n\n"
            f"Statement A (new):\n{item.text}\n\n"
            f"Statement B (existing ticket {search_result.item_id}):\n{existing_text}\n\n"
            "If they contradict each other, respond with:\n"
            "CONTRADICTION: <brief description of the contradiction>\n\n"
            "If they do not contradict, respond with:\n"
            "NO CONTRADICTION"
        )

        response = await asyncio.to_thread(
            self._llm_tool.generate,
            prompt,
            "You are a precise analyst detecting contradictions between requirements.",
        )

        if response.strip().startswith("CONTRADICTION:"):
            description = response.strip().removeprefix("CONTRADICTION:").strip()
            return True, description if description else "Contradicting statements detected"

        return False, None
