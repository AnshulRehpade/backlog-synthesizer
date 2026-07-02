"""Concrete implementation of the DocumentParsingTool interface.

Uses PyMuPDF (fitz) for PDF-to-text conversion and whitespace-based
tokenization for text chunking. All library-specific exceptions are
translated to interface-defined ToolError subtypes before propagating.
"""

from __future__ import annotations

import fitz  # PyMuPDF

from backlog_synthesizer.tools.errors import PermanentToolError, TransientToolError


class PyMuPDFDocumentParser:
    """DocumentParsingTool implementation backed by PyMuPDF.

    Satisfies the DocumentParsingTool Protocol defined in
    backlog_synthesizer.tools.interfaces.
    """

    def pdf_to_text(self, content: bytes) -> str:
        """Convert PDF binary content to plain text.

        Uses PyMuPDF to extract text page-by-page, preserving paragraph
        boundaries by joining pages with double newlines.

        Args:
            content: Raw PDF file bytes.

        Returns:
            Extracted text content preserving paragraph boundaries.

        Raises:
            PermanentToolError: If the PDF is malformed or cannot be parsed.
            TransientToolError: If a transient I/O error occurs.
        """
        if not content:
            raise PermanentToolError(
                "Cannot parse empty PDF content", original_error=None
            )

        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as e:
            raise PermanentToolError(
                f"Failed to open PDF document: {e}", original_error=e
            ) from e

        try:
            pages: list[str] = []
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    pages.append(page_text.strip())
            return "\n\n".join(pages)
        except MemoryError as e:
            raise TransientToolError(
                f"Insufficient memory to process PDF: {e}", original_error=e
            ) from e
        except Exception as e:
            raise PermanentToolError(
                f"Error extracting text from PDF: {e}", original_error=e
            ) from e
        finally:
            doc.close()

    def chunk_text(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        """Split text into overlapping chunks using whitespace tokenization.

        Tokens are defined by splitting on whitespace. Each chunk contains
        at most `max_tokens` tokens, and consecutive chunks share `overlap`
        tokens.

        Args:
            text: The input text to chunk.
            max_tokens: Maximum number of tokens per chunk.
            overlap: Number of overlapping tokens between consecutive chunks.

        Returns:
            List of text chunks. Returns an empty list if text is empty.

        Raises:
            PermanentToolError: If parameters are invalid.
        """
        if max_tokens <= 0:
            raise PermanentToolError(
                f"max_tokens must be positive, got {max_tokens}", original_error=None
            )
        if overlap < 0:
            raise PermanentToolError(
                f"overlap must be non-negative, got {overlap}", original_error=None
            )
        if overlap >= max_tokens:
            raise PermanentToolError(
                f"overlap ({overlap}) must be less than max_tokens ({max_tokens})",
                original_error=None,
            )

        tokens = text.split()
        if not tokens:
            return []

        chunks: list[str] = []
        step = max_tokens - overlap
        start = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(" ".join(chunk_tokens))
            start += step

        return chunks
