"""Few-shot example store for dynamic prompt construction.

Indexes golden dataset examples into a vector store and retrieves
the most semantically similar examples for a given input text.
"""

import json
import logging
from pathlib import Path
from typing import Any

from backlog_synthesizer.tools.interfaces import EmbeddingTool, SearchResult, VectorSearchTool

logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path("data/golden_dataset/golden_entries.json")
SIMILARITY_THRESHOLD = 0.5  # Below this, examples are considered unhelpful


class FewShotExample:
    """A single few-shot example with its metadata."""

    def __init__(
        self,
        id: str,
        transcript: str,
        description: str,
        tags: list[str],
        score: float = 0.0,
    ):
        self.id = id
        self.transcript = transcript
        self.description = description
        self.tags = tags
        self.score = score


class FewShotStore:
    """Indexes golden examples and retrieves relevant few-shot examples.

    Uses embedding similarity to find the most relevant examples for a given
    input text. Results are filtered by SIMILARITY_THRESHOLD to avoid
    injecting unhelpful examples.
    """

    def __init__(
        self,
        embedding_tool: EmbeddingTool,
        vector_search_tool: VectorSearchTool,
    ):
        self._embedding_tool = embedding_tool
        self._vector_search_tool = vector_search_tool
        self._indexed = False
        self._entries: dict[str, dict[str, Any]] = {}  # id -> full entry data

    def index_golden_dataset(self, dataset_path: Path | None = None) -> int:
        """Index all golden dataset examples into the vector store.

        Args:
            dataset_path: Path to golden_entries.json. Uses default if None.

        Returns:
            Number of entries indexed.
        """
        path = dataset_path or GOLDEN_DATASET_PATH
        if not path.exists():
            logger.warning("Golden dataset not found at %s", path)
            return 0

        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load golden dataset: %s", e)
            return 0

        count = 0
        for entry in entries:
            entry_id = entry.get("id", "")
            transcript = entry.get("transcript", "")
            if not transcript or not entry_id:
                continue

            try:
                embedding = self._embedding_tool.generate_embedding(transcript)
                self._vector_search_tool.store(
                    item_id=f"few_shot_{entry_id}",
                    embedding=embedding,
                    metadata={
                        "golden_id": entry_id,
                        "description": entry.get("description", ""),
                        "tags": ",".join(entry.get("tags", [])),
                        "transcript": transcript[:500],  # store truncated for retrieval
                    },
                )
                self._entries[entry_id] = entry
                count += 1
            except Exception as e:
                logger.warning("Failed to index golden entry %s: %s", entry_id, e)

        self._indexed = True
        logger.info("Indexed %d golden entries for few-shot retrieval", count)
        return count

    def get_similar_parser_examples(
        self, text: str, top_k: int = 2
    ) -> list[FewShotExample]:
        """Retrieve the most relevant extraction examples for a given text.

        Args:
            text: Input text to find similar examples for.
            top_k: Maximum number of examples to return.

        Returns:
            List of FewShotExample objects with score >= SIMILARITY_THRESHOLD.
            Empty list if store is not indexed or no good matches found.
        """
        if not self._indexed:
            return []

        try:
            embedding = self._embedding_tool.generate_embedding(text)
            results = self._vector_search_tool.query_similar(embedding, top_k)
        except Exception as e:
            logger.warning("Few-shot retrieval failed: %s", e)
            return []

        examples = []
        for result in results:
            if result.score < SIMILARITY_THRESHOLD:
                continue
            golden_id = result.metadata.get("golden_id", "")
            entry = self._entries.get(golden_id)
            if entry:
                examples.append(
                    FewShotExample(
                        id=golden_id,
                        transcript=entry.get("transcript", ""),
                        description=entry.get("description", ""),
                        tags=entry.get("tags", []),
                        score=result.score,
                    )
                )

        return examples

    def get_similar_story_examples(
        self, item_text: str, top_k: int = 2
    ) -> list[FewShotExample]:
        """Retrieve the most relevant story writing examples.

        Uses the same retrieval mechanism as parser examples —
        finds golden entries whose transcript is semantically similar
        to the item text being converted to a story.

        Args:
            item_text: The extracted item text to find similar examples for.
            top_k: Maximum number of examples to return.

        Returns:
            List of FewShotExample objects with score >= SIMILARITY_THRESHOLD.
        """
        # Same implementation — similarity is computed against transcripts
        return self.get_similar_parser_examples(item_text, top_k)
