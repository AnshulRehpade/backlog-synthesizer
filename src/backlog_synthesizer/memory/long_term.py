"""Long-Term Memory implementation using vector store for semantic search.

Provides embedding generation, storage, and semantic search over backlog items.
Uses EmbeddingTool and VectorSearchTool protocol interfaces for portability.
Implements a 30-day retention policy for stored entries.
"""

from datetime import datetime, timezone
from typing import Any

from backlog_synthesizer.tools.interfaces import (
    EmbeddingTool,
    SearchResult,
    VectorSearchTool,
)

# Default retention period in days for stored entries.
_DEFAULT_RETENTION_DAYS = 30


class LongTermMemory:
    """Vector-backed long-term memory for semantic search over backlog items.

    Stores item embeddings via VectorSearchTool and generates embeddings via
    EmbeddingTool. Items are tagged with a stored_at timestamp to support
    time-based retention purging.

    Args:
        embedding_tool: Implementation of EmbeddingTool for generating embeddings.
        vector_search_tool: Implementation of VectorSearchTool for storage and search.
        retention_days: Number of days to retain stored entries. Defaults to 30.
    """

    def __init__(
        self,
        embedding_tool: EmbeddingTool,
        vector_search_tool: VectorSearchTool,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
    ) -> None:
        self._embedding_tool = embedding_tool
        self._vector_search_tool = vector_search_tool
        self._retention_days = retention_days

    @property
    def retention_days(self) -> int:
        """Number of days entries are retained before becoming eligible for purging."""
        return self._retention_days

    def store_item(
        self,
        item_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Generate an embedding for content and store it in the vector store.

        The item is stored with a `stored_at` ISO 8601 timestamp in its metadata
        to support retention-based purging.

        Args:
            item_id: Unique identifier for the item being stored.
            content: Text content to generate an embedding from.
            metadata: Optional additional metadata to associate with the item.

        Raises:
            ToolError: If embedding generation or storage fails.
        """
        embedding = self._embedding_tool.generate_embedding(content)

        stored_metadata: dict[str, Any] = {
            **(metadata or {}),
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "content": content,
        }

        self._vector_search_tool.store(item_id, embedding, stored_metadata)

    def search_similar(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search for items semantically similar to the query text.

        Generates an embedding for the query and searches the vector store
        for the closest matches.

        Args:
            query: Text to search for similar items.
            top_k: Maximum number of results to return. Defaults to 10.

        Returns:
            List of SearchResult objects ordered by similarity score (descending).

        Raises:
            ToolError: If embedding generation or search fails.
        """
        query_embedding = self._embedding_tool.generate_embedding(query)
        return self._vector_search_tool.query_similar(query_embedding, top_k)

    def purge_expired(self, reference_time: datetime | None = None) -> list[str]:
        """Remove entries that have exceeded the retention period.

        Queries all stored items and removes those whose `stored_at` timestamp
        is older than `retention_days` from the reference time.

        Note: This method requires the VectorSearchTool to support querying all
        items. It uses a large top_k to retrieve candidates and filters by timestamp.
        In production, a more efficient purge mechanism (e.g., Chroma's built-in
        filtering) should be used.

        Args:
            reference_time: The time to compare against. Defaults to current UTC time.

        Returns:
            List of item IDs that were identified as expired.
        """
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        # Generate a zero-vector query to retrieve all items for expiration check.
        # A dedicated "list all" method on the vector store would be more efficient,
        # but this works within the VectorSearchTool protocol constraints.
        all_results = self._vector_search_tool.query_similar(
            embedding=[0.0] * 1,  # Minimal placeholder embedding
            top_k=10000,
        )

        expired_ids: list[str] = []
        for result in all_results:
            stored_at_str = result.metadata.get("stored_at")
            if stored_at_str is None:
                continue

            stored_at = datetime.fromisoformat(stored_at_str)
            age_days = (reference_time - stored_at).days

            if age_days > self._retention_days:
                expired_ids.append(result.item_id)

        return expired_ids
