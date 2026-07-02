"""Main entry point for the Backlog Synthesizer pipeline.

Wires all components together: PipelineConfig, MemoryEngine, all agents,
and OrchestratorAgent. Exposes a `run` function that accepts file paths
and returns serialized StoryOutput JSON.

Requirements: 2.1, 2.2, 2.3, 2.4, 9.7
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root (before any config reads)
load_dotenv()

from backlog_synthesizer.agents.gap_detection import GapDetectionAgent
from backlog_synthesizer.agents.orchestrator import OrchestratorAgent, SessionResult
from backlog_synthesizer.agents.parser import ParserAgent
from backlog_synthesizer.agents.story_writer import StoryWriterAgent
from backlog_synthesizer.config import PipelineConfig
from backlog_synthesizer.memory.audit_log import AuditLog
from backlog_synthesizer.memory.engine import MemoryEngine
from backlog_synthesizer.memory.long_term import LongTermMemory
from backlog_synthesizer.memory.short_term import ShortTermMemory
from backlog_synthesizer.models.inputs import (
    BacklogTicket,
    DocumentType,
    InputDocument,
    SessionInputs,
)

logger = logging.getLogger(__name__)

# File extension to DocumentType mapping
_EXTENSION_MAP: dict[str, DocumentType] = {
    ".txt": DocumentType.TRANSCRIPT_TXT,
    ".md": DocumentType.TRANSCRIPT_MD,
    ".pdf": DocumentType.TRANSCRIPT_PDF,
    ".html": DocumentType.ARCHITECTURE_HTML,
    ".json": DocumentType.BACKLOG_JSON,
}


def _determine_document_type(file_path: Path) -> DocumentType:
    """Determine the DocumentType from a file's extension.

    Args:
        file_path: Path to the file.

    Returns:
        The corresponding DocumentType.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = file_path.suffix.lower()
    if ext not in _EXTENSION_MAP:
        raise ValueError(
            f"Unsupported file extension '{ext}' for file '{file_path.name}'. "
            f"Supported extensions: {', '.join(_EXTENSION_MAP.keys())}"
        )
    return _EXTENSION_MAP[ext]


def create_pipeline(config: PipelineConfig | None = None) -> OrchestratorAgent:
    """Create a fully-wired OrchestratorAgent with all dependencies.

    Instantiates PipelineConfig, tool implementations, MemoryEngine,
    and all agents. Returns the orchestrator ready to run sessions.

    Args:
        config: Optional PipelineConfig. If None, loads from environment.

    Returns:
        A fully configured OrchestratorAgent instance.
    """
    if config is None:
        config = PipelineConfig.from_env()

    # Create tool implementations from config
    parsing_tool = config.create_document_parser()
    embedding_tool = config.create_embedding_tool()
    vector_search_tool = config.create_vector_search_tool()
    llm_tool = config.create_llm_tool()

    # Create Memory Engine components
    short_term = ShortTermMemory()
    audit_log = AuditLog()
    long_term = LongTermMemory(
        embedding_tool=embedding_tool,
        vector_search_tool=vector_search_tool,
    )
    memory = MemoryEngine(
        short_term=short_term,
        long_term=long_term,
        audit_log=audit_log,
    )

    # Create agents
    parser = ParserAgent(parsing_tool=parsing_tool, llm_tool=llm_tool)
    gap_detector = GapDetectionAgent(
        embedding_tool=embedding_tool,
        vector_search_tool=vector_search_tool,
        llm_tool=llm_tool,
    )
    story_writer = StoryWriterAgent(llm_tool=llm_tool)

    # Create orchestrator
    orchestrator = OrchestratorAgent(
        parser=parser,
        gap_detector=gap_detector,
        story_writer=story_writer,
        memory=memory,
    )

    return orchestrator


def _build_session_inputs(file_paths: list[str | Path]) -> SessionInputs:
    """Build SessionInputs from a list of file paths.

    Reads file contents, determines document types from extensions,
    and parses JSON files as backlog tickets.

    Args:
        file_paths: List of file paths to process.

    Returns:
        A SessionInputs instance with documents and backlog tickets.

    Raises:
        FileNotFoundError: If a file does not exist.
        ValueError: If a file has an unsupported extension.
    """
    session_id = str(uuid.uuid4())
    documents: list[InputDocument] = []
    backlog_tickets: list[BacklogTicket] = []

    for file_path in file_paths:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        doc_type = _determine_document_type(path)
        content = path.read_bytes()

        if doc_type == DocumentType.BACKLOG_JSON:
            # Parse JSON file as backlog tickets
            text_content = content.decode("utf-8")
            raw_tickets = json.loads(text_content)
            if isinstance(raw_tickets, list):
                for raw_ticket in raw_tickets:
                    try:
                        ticket = BacklogTicket.model_validate(raw_ticket)
                        backlog_tickets.append(ticket)
                    except Exception as e:
                        logger.warning(
                            "Failed to validate backlog ticket from %s: %s",
                            path.name, e,
                        )
            else:
                logger.warning(
                    "Expected a JSON array in %s, got %s",
                    path.name, type(raw_tickets).__name__,
                )
        else:
            documents.append(
                InputDocument(
                    filename=path.name,
                    document_type=doc_type,
                    content=content,
                    size_bytes=len(content),
                )
            )

    return SessionInputs(
        session_id=session_id,
        documents=documents,
        backlog_tickets=backlog_tickets,
    )


async def run_async(
    file_paths: list[str | Path],
    config: PipelineConfig | None = None,
) -> str:
    """Run the Backlog Synthesizer pipeline asynchronously.

    Creates the pipeline, builds session inputs from the provided file paths,
    runs the orchestrator, and returns the serialized StoryOutput JSON.

    Args:
        file_paths: List of file paths to process.
        config: Optional PipelineConfig. If None, loads from environment.

    Returns:
        A JSON string of the StoryOutput, or an error JSON if the pipeline fails.
    """
    orchestrator = create_pipeline(config)
    inputs = _build_session_inputs(file_paths)

    result: SessionResult = await orchestrator.run_session(inputs)

    if result.output is not None:
        return result.output.model_dump_json(indent=2)
    else:
        # Return error information as JSON
        error_response = {
            "session_id": result.session_id,
            "status": result.status,
            "errors": result.errors,
        }
        return json.dumps(error_response, indent=2, default=str)


def run(
    file_paths: list[str | Path],
    config: PipelineConfig | None = None,
) -> str:
    """Run the Backlog Synthesizer pipeline synchronously.

    Convenience wrapper around `run_async` for non-async callers.

    Args:
        file_paths: List of file paths to process.
        config: Optional PipelineConfig. If None, loads from environment.

    Returns:
        A JSON string of the StoryOutput, or an error JSON if the pipeline fails.
    """
    return asyncio.run(run_async(file_paths, config))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m backlog_synthesizer.main <file1> [file2] ...")
        print("")
        print("Supported file types:")
        print("  .txt, .md  - Meeting transcripts")
        print("  .pdf       - PDF transcripts")
        print("  .html      - Architecture documents (wiki export)")
        print("  .json      - Existing backlog tickets")
        sys.exit(1)

    output = run(sys.argv[1:])
    print(output)
