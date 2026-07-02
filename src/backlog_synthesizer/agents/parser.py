"""Parser Agent for document ingestion, chunking, and information extraction."""

import asyncio
import json
import logging
from html.parser import HTMLParser

from backlog_synthesizer.models.extraction import (
    DocumentError,
    ExtractedItem,
    ExtractionResult,
    TextChunk,
)
from backlog_synthesizer.models.inputs import DocumentType, InputDocument
from backlog_synthesizer.tools.errors import ToolError
from backlog_synthesizer.tools.interfaces import DocumentParsingTool, LLMGenerationTool

logger = logging.getLogger(__name__)


class _HTMLToStructuredText(HTMLParser):
    """Converts HTML to structured text preserving heading hierarchy.

    Headings (h1-h6) are rendered as markdown-style prefixes:
      h1 → "# Heading"
      h2 → "## Heading"
      ...
      h6 → "###### Heading"

    All other HTML tags are stripped and their text content preserved.
    """

    def __init__(self) -> None:
        super().__init__()
        self._output: list[str] = []
        self._current_text: list[str] = []
        self._in_heading: int = 0  # 0 = not in heading, 1-6 = heading level

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("h1", "h2", "h3", "h4", "h5", "h6"):
            # Flush any pending text before the heading
            self._flush_text()
            self._in_heading = int(tag_lower[1])

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag_lower[1])
            if self._in_heading == level:
                heading_text = "".join(self._current_text).strip()
                if heading_text:
                    prefix = "#" * level
                    self._output.append(f"{prefix} {heading_text}")
                self._current_text = []
                self._in_heading = 0
        elif tag_lower in ("p", "div", "br", "li", "tr"):
            # Block-level elements get newlines
            self._flush_text()

    def handle_data(self, data: str) -> None:
        self._current_text.append(data)

    def _flush_text(self) -> None:
        text = "".join(self._current_text).strip()
        if text:
            self._output.append(text)
        self._current_text = []

    def get_result(self) -> str:
        """Return the final structured text after parsing is complete."""
        self._flush_text()
        return "\n\n".join(self._output)


class ParserAgent:
    """Agent responsible for ingesting documents, chunking text, and extracting
    decisions, pain points, and feature requests using an LLM.
    """

    def __init__(self, parsing_tool: DocumentParsingTool, llm_tool: LLMGenerationTool):
        self._parsing_tool = parsing_tool
        self._llm_tool = llm_tool

    async def parse_documents(self, documents: list[InputDocument]) -> ExtractionResult:
        """Parse all input documents and extract structured items.

        When multiple documents are provided, they are processed concurrently
        using asyncio.gather() for improved throughput.

        Args:
            documents: List of input documents to process.

        Returns:
            ExtractionResult containing extracted items and any errors.
        """
        all_items: list[ExtractedItem] = []
        all_errors: list[DocumentError] = []

        # Ingest documents (synchronous I/O — chunk and classify)
        doc_tasks: list[tuple[InputDocument, str | None, DocumentError | None]] = []
        for doc in documents:
            text, error = self._ingest_document(doc)
            doc_tasks.append((doc, text, error))

        # Collect errors from ingestion
        processable: list[tuple[InputDocument, str]] = []
        for doc, text, error in doc_tasks:
            if error is not None:
                all_errors.append(error)
            elif text is not None:
                processable.append((doc, text))

        # Extract from all documents concurrently
        if processable:
            extraction_coros = [
                self._extract_document_async(doc, text) for doc, text in processable
            ]
            results = await asyncio.gather(*extraction_coros, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Document extraction failed: %s", result)
                    continue
                all_items.extend(result)

        metadata: dict = {}
        if not all_items:
            metadata["note"] = "Document processed but yielded no extractable items"

        return ExtractionResult(items=all_items, errors=all_errors, metadata=metadata)

    async def _extract_document_async(
        self, doc: InputDocument, text: str
    ) -> list[ExtractedItem]:
        """Extract items from a single document asynchronously.

        Chunks the text and runs LLM extraction in a thread pool to avoid
        blocking the event loop.

        Args:
            doc: The input document metadata.
            text: The ingested text content.

        Returns:
            List of extracted items from this document.
        """
        chunks = self._chunk_text(text)

        if self._is_transcript(doc.document_type):
            return await asyncio.to_thread(self._extract_from_transcript, chunks)
        elif doc.document_type == DocumentType.ARCHITECTURE_HTML:
            return await asyncio.to_thread(self._extract_from_architecture, chunks)
        else:
            return []

    def _is_transcript(self, doc_type: DocumentType) -> bool:
        """Check if a document type is a meeting transcript."""
        return doc_type in (
            DocumentType.TRANSCRIPT_TXT,
            DocumentType.TRANSCRIPT_MD,
            DocumentType.TRANSCRIPT_PDF,
        )

    def _extract_from_transcript(self, chunks: list[TextChunk]) -> list[ExtractedItem]:
        """Extract decisions, pain points, and feature requests from transcript chunks.

        For each chunk, calls the LLM to identify and extract structured items.
        Gracefully handles JSON parsing errors and LLM failures.

        Args:
            chunks: List of text chunks from a transcript document.

        Returns:
            List of extracted items with confidence scores and source metadata.
        """
        items: list[ExtractedItem] = []

        system_prompt = (
            "You are an expert at analyzing meeting transcripts. "
            "Extract decisions, pain points, and feature requests from the text. "
            "Return a JSON array of objects. Each object must have these fields:\n"
            '- "item_type": one of "decision", "pain_point", "feature_request"\n'
            '- "text": the extracted text describing the item\n'
            '- "confidence": a float between 0.0 and 1.0 indicating confidence\n'
            '- "char_offset": integer character offset within the chunk where the item starts\n'
            '- "stakeholder": the person or role affected/requesting (null if not identifiable)\n'
            "If no items are found, return an empty array: []\n"
            "Return ONLY valid JSON, no additional text."
        )

        for chunk in chunks:
            prompt = f"Analyze this meeting transcript chunk and extract items:\n\n{chunk.text}"
            try:
                response = self._llm_tool.generate(prompt, system_prompt=system_prompt)
            except ToolError:
                logger.warning(
                    "LLM generation failed for chunk %d, skipping.", chunk.index
                )
                continue

            # Strip markdown code fences if present
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                # Remove first line (```json) and last line (```)
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response = "\n".join(lines)

            try:
                parsed = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "LLM response for chunk %d was not valid JSON, skipping.", chunk.index
                )
                continue

            if not isinstance(parsed, list):
                logger.warning(
                    "LLM response for chunk %d was not a JSON array, skipping.", chunk.index
                )
                continue

            for entry in parsed:
                if not isinstance(entry, dict):
                    continue
                try:
                    item = ExtractedItem(
                        item_type=entry.get("item_type", "decision"),
                        text=entry.get("text", ""),
                        source_chunk_index=chunk.index,
                        char_offset=entry.get("char_offset"),
                        confidence=float(entry.get("confidence", 0.5)),
                        stakeholder=entry.get("stakeholder"),
                    )
                    # Skip items with empty text
                    if not item.text:
                        continue
                    items.append(item)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "Failed to create ExtractedItem from chunk %d entry: %s",
                        chunk.index,
                        e,
                    )
                    continue

        return items

    def _extract_from_architecture(self, chunks: list[TextChunk]) -> list[ExtractedItem]:
        """Extract technical constraints and architectural decisions from architecture doc chunks.

        For each chunk, calls the LLM to identify constraints, decisions, and principles.
        Gracefully handles JSON parsing errors and LLM failures.

        Args:
            chunks: List of text chunks from an architecture document.

        Returns:
            List of extracted items with type classification and section heading.
        """
        items: list[ExtractedItem] = []

        system_prompt = (
            "You are an expert at analyzing architecture documents. "
            "Extract technical constraints and architectural decisions from the text. "
            "Return a JSON array of objects. Each object must have these fields:\n"
            '- "item_type": always "constraint" for this document type\n'
            '- "text": the extracted constraint or decision text\n'
            '- "confidence": a float between 0.0 and 1.0 indicating confidence\n'
            '- "char_offset": integer character offset within the chunk where the item starts\n'
            '- "section_heading": the section heading this item belongs to (null if unknown)\n'
            '- "type_classification": one of "constraint", "decision", "principle"\n'
            "If no items are found, return an empty array: []\n"
            "Return ONLY valid JSON, no additional text."
        )

        for chunk in chunks:
            prompt = (
                f"Analyze this architecture document chunk and extract "
                f"constraints and decisions:\n\n{chunk.text}"
            )
            try:
                response = self._llm_tool.generate(prompt, system_prompt=system_prompt)
            except ToolError:
                logger.warning(
                    "LLM generation failed for architecture chunk %d, skipping.",
                    chunk.index,
                )
                continue

            # Strip markdown code fences if present
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response = "\n".join(lines)

            try:
                parsed = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "LLM response for architecture chunk %d was not valid JSON, skipping.",
                    chunk.index,
                )
                continue

            if not isinstance(parsed, list):
                logger.warning(
                    "LLM response for architecture chunk %d was not a JSON array, skipping.",
                    chunk.index,
                )
                continue

            for entry in parsed:
                if not isinstance(entry, dict):
                    continue
                try:
                    item = ExtractedItem(
                        item_type="constraint",
                        text=entry.get("text", ""),
                        source_chunk_index=chunk.index,
                        char_offset=entry.get("char_offset"),
                        confidence=float(entry.get("confidence", 0.5)),
                        section_heading=entry.get("section_heading"),
                        type_classification=entry.get("type_classification"),
                    )
                    # Skip items with empty text
                    if not item.text:
                        continue
                    items.append(item)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "Failed to create ExtractedItem from architecture chunk %d entry: %s",
                        chunk.index,
                        e,
                    )
                    continue

        return items

    def _ingest_text(self, doc: InputDocument) -> str:
        """Ingest a text or markdown document by decoding bytes to UTF-8.

        Args:
            doc: Input document with content as raw bytes.

        Returns:
            The decoded text content.

        Raises:
            UnicodeDecodeError: If the content cannot be decoded as UTF-8.
        """
        return doc.content.decode("utf-8")

    def _ingest_pdf(self, doc: InputDocument) -> str:
        """Ingest a PDF document by converting to text via DocumentParsingTool.

        Args:
            doc: Input document with PDF content as raw bytes.

        Returns:
            Extracted text preserving paragraph boundaries.

        Raises:
            ToolError: If PDF parsing fails.
        """
        return self._parsing_tool.pdf_to_text(doc.content)

    def _ingest_html(self, doc: InputDocument) -> str:
        """Ingest an HTML wiki export document by stripping markup and preserving heading hierarchy.

        Heading elements (h1-h6) are converted to markdown-style prefixes.
        All other HTML tags are stripped. Block-level elements produce paragraph breaks.

        Args:
            doc: Input document with HTML content as raw bytes.

        Returns:
            Structured text with heading hierarchy preserved.

        Raises:
            UnicodeDecodeError: If the content cannot be decoded as UTF-8.
        """
        html_text = doc.content.decode("utf-8")
        parser = _HTMLToStructuredText()
        parser.feed(html_text)
        parser.close()
        return parser.get_result()

    def _ingest_document(self, doc: InputDocument) -> tuple[str | None, DocumentError | None]:
        """Route document to the appropriate ingestion method based on type.

        Args:
            doc: The input document to ingest.

        Returns:
            A tuple of (text_content, error). On success, text_content is the
            extracted text and error is None. On failure, text_content is None
            and error contains details about the failure.
        """
        try:
            if doc.document_type in (DocumentType.TRANSCRIPT_TXT, DocumentType.TRANSCRIPT_MD):
                text = self._ingest_text(doc)
            elif doc.document_type == DocumentType.TRANSCRIPT_PDF:
                text = self._ingest_pdf(doc)
            elif doc.document_type == DocumentType.ARCHITECTURE_HTML:
                text = self._ingest_html(doc)
            else:
                return None, DocumentError(
                    filename=doc.filename,
                    reason=f"Unsupported document type: {doc.document_type.value}",
                )
            return text, None
        except UnicodeDecodeError as e:
            return None, DocumentError(
                filename=doc.filename,
                reason=f"Failed to decode document as UTF-8: {e.reason}",
                byte_offset=e.start,
            )
        except ToolError as e:
            return None, DocumentError(
                filename=doc.filename,
                reason=f"Document parsing tool error: {str(e)}",
            )
        except Exception as e:
            return None, DocumentError(
                filename=doc.filename,
                reason=f"Unexpected error during ingestion: {str(e)}",
            )

    def _chunk_text(
        self, text: str, max_tokens: int = 2000, overlap: int = 200
    ) -> list[TextChunk]:
        """Split text into overlapping chunks using whitespace-based tokenization.

        Each chunk contains at most `max_tokens` tokens. Consecutive chunks share
        exactly `overlap` tokens so that no information is lost at boundaries.
        Reconstructing the original text by removing overlapping regions from
        consecutive chunks reproduces the input exactly.

        Args:
            text: The input text to chunk.
            max_tokens: Maximum number of tokens (whitespace-delimited words) per chunk.
            overlap: Number of tokens shared between consecutive chunks.

        Returns:
            A list of TextChunk objects. Returns an empty list for empty input.
            Returns a single chunk if the text has fewer than max_tokens tokens.
        """
        if not text:
            return []

        tokens = text.split()
        total_tokens = len(tokens)

        if total_tokens == 0:
            return []

        # If the entire text fits in one chunk, return it as-is
        if total_tokens <= max_tokens:
            return [
                TextChunk(
                    index=0,
                    text=text,
                    token_count=total_tokens,
                )
            ]

        chunks: list[TextChunk] = []
        # Step size is how far we advance the start position each iteration
        step = max_tokens - overlap
        index = 0
        start = 0

        while start < total_tokens:
            end = min(start + max_tokens, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)

            chunks.append(
                TextChunk(
                    index=index,
                    text=chunk_text,
                    token_count=len(chunk_tokens),
                )
            )

            # If we've reached the end, stop
            if end == total_tokens:
                break

            start += step
            index += 1

        return chunks
