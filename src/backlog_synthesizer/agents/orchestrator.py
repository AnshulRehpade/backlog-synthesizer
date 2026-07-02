"""Orchestrator Agent for the Backlog Synthesizer system.

Coordinates sub-agents in sequence (Parser → Gap Detection → Story Writer),
manages session state, validates backlog tickets, and stores intermediate
results in the Memory Engine.

Requirements: 2.1, 2.2, 2.3, 2.4, 1.4, 1.6, 2.5, 2.6, 7.1, 7.2, 7.3, 7.5, 7.6
"""

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import Any, TypeVar

from pydantic import ValidationError

from backlog_synthesizer.agents.errors import PipelineHaltError, RetryExhaustedError
from backlog_synthesizer.agents.gap_detection import GapDetectionAgent
from backlog_synthesizer.agents.parser import ParserAgent
from backlog_synthesizer.agents.story_writer import StoryWriterAgent
from backlog_synthesizer.memory.engine import MemoryEngine
from backlog_synthesizer.models.extraction import ExtractionResult
from backlog_synthesizer.models.gap_detection import GapReport
from backlog_synthesizer.models.inputs import (
    BacklogTicket,
    DocumentType,
    SessionInputs,
)
from backlog_synthesizer.models.memory import AuditEntry, SessionState
from backlog_synthesizer.models.output import StoryOutput
from backlog_synthesizer.tools.errors import PermanentToolError, TransientToolError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SessionResult:
    """Result of a pipeline session execution.

    Attributes:
        session_id: The session identifier.
        status: One of "completed", "partial_failure", "permanent_failure".
        output: The final StoryOutput if story generation succeeded, else None.
        session_state: The full session state with intermediate results.
        errors: List of error dicts describing any failures.
    """

    def __init__(
        self,
        session_id: str,
        status: str,
        output: StoryOutput | None = None,
        session_state: SessionState | None = None,
        errors: list[dict] | None = None,
    ) -> None:
        self.session_id = session_id
        self.status = status
        self.output = output
        self.session_state = session_state
        self.errors = errors or []


class OrchestratorAgent:
    """Central coordination agent that manages the processing pipeline.

    Receives input, coordinates sub-agents in sequence (Parser → Gap Detection
    → Story Writer), validates backlog tickets, stores intermediate results in
    the Memory Engine, and produces a final SessionResult.

    Requirements: 2.1, 2.2, 2.3, 2.4, 1.4, 1.6
    """

    def __init__(
        self,
        parser: ParserAgent,
        gap_detector: GapDetectionAgent,
        story_writer: StoryWriterAgent,
        memory: MemoryEngine,
    ) -> None:
        """Initialize OrchestratorAgent with sub-agents and memory engine.

        Args:
            parser: The Parser Agent for document ingestion and extraction.
            gap_detector: The Gap Detection Agent for duplicate/conflict analysis.
            story_writer: The Story Writer Agent for user story generation.
            memory: The Memory Engine for session state and persistence.
        """
        self._parser = parser
        self._gap_detector = gap_detector
        self._story_writer = story_writer
        self._memory = memory

    async def run_session(self, inputs: SessionInputs) -> SessionResult:
        """Execute the full pipeline for a set of inputs.

        Pipeline sequence:
        1. Validate backlog tickets and load valid ones into Long-Term Memory
        2. Invoke Parser Agent with non-backlog documents
        3. Store extraction results in Memory Engine
        4. Invoke Gap Detection Agent with extracted items
        5. Store gap report in Memory Engine
        6. Invoke Story Writer Agent with deduplicated items
        7. Store final output in Memory Engine

        Args:
            inputs: The session inputs containing documents and backlog tickets.

        Returns:
            A SessionResult with the pipeline outcome.
        """
        session_id = inputs.session_id

        # Create session state
        session_state = SessionState(
            session_id=session_id,
            created_at=datetime.now(timezone.utc),
            status="in_progress",
        )

        # Store initial session state
        self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())

        errors: list[dict] = []

        # Step 0: Validate backlog tickets
        valid_tickets = self._validate_backlog_tickets(session_id, inputs.backlog_tickets, errors)

        # Load valid tickets into Long-Term Memory for comparison
        if valid_tickets:
            ticket_items = [
                {
                    "item_id": ticket.id,
                    "content": f"{ticket.title} {ticket.description}",
                    "status": ticket.status,
                    "tags": ticket.tags,
                }
                for ticket in valid_tickets
            ]
            self._memory.store_for_search(session_id, ticket_items)

        # Step 1: Invoke Parser Agent (Requirement 2.1)
        # Filter to non-backlog documents only
        non_backlog_docs = [
            doc for doc in inputs.documents
            if doc.document_type != DocumentType.BACKLOG_JSON
        ]

        start_time = time.time()
        try:
            extraction_result = await self._invoke_with_retry(
                self._parser.parse_documents, non_backlog_docs
            )
        except PipelineHaltError as e:
            logger.error("Parser Agent halted (permanent error): %s", e)
            duration_ms = int((time.time() - start_time) * 1000)
            self._log_agent_action(
                session_id, "ParserAgent",
                f"Documents: {len(non_backlog_docs)}",
                f"Permanent error: {str(e.original_error)[:200]}",
                duration_ms,
            )
            errors.append({"step": "parser", "error": str(e.original_error)})
            session_state.status = "permanent_failure"
            session_state.errors = errors
            self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())
            return SessionResult(
                session_id=session_id,
                status="permanent_failure",
                session_state=session_state,
                errors=errors,
            )
        except RetryExhaustedError as e:
            logger.error("Parser Agent retries exhausted: %s", e)
            duration_ms = int((time.time() - start_time) * 1000)
            self._log_agent_action(
                session_id, "ParserAgent",
                f"Documents: {len(non_backlog_docs)}",
                f"Retries exhausted ({e.attempts} attempts): {str(e.original_error)[:200]}",
                duration_ms,
            )
            errors.append({"step": "parser", "error": str(e.original_error)})
            session_state.status = "partial_failure"
            session_state.errors = errors
            self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())
            return SessionResult(
                session_id=session_id,
                status="partial_failure",
                session_state=session_state,
                errors=errors,
            )
        except Exception as e:
            logger.error("Parser Agent failed: %s", e)
            duration_ms = int((time.time() - start_time) * 1000)
            self._log_agent_action(
                session_id, "ParserAgent",
                f"Documents: {len(non_backlog_docs)}",
                f"Error: {str(e)[:200]}",
                duration_ms,
            )
            errors.append({"step": "parser", "error": str(e)})
            session_state.status = "partial_failure"
            session_state.errors = errors
            self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())
            return SessionResult(
                session_id=session_id,
                status="partial_failure",
                session_state=session_state,
                errors=errors,
            )

        duration_ms = int((time.time() - start_time) * 1000)
        self._log_agent_action(
            session_id, "ParserAgent",
            f"Documents: {len(non_backlog_docs)}",
            f"Extracted {len(extraction_result.items)} items, {len(extraction_result.errors)} errors",
            duration_ms,
        )

        # Store extraction results (Requirement 2.4)
        session_state.extraction_result = extraction_result
        self._memory.store_intermediate(session_id, "extraction_result", extraction_result.model_dump())

        # NOTE: Extracted items are stored in Long-Term Memory AFTER gap detection
        # to avoid self-matching during duplicate analysis.

        # Step 2: Invoke Gap Detection Agent (Requirement 2.2)
        start_time = time.time()
        try:
            gap_report = await self._invoke_with_retry(
                self._gap_detector.analyze_gaps, extraction_result.items
            )
        except PipelineHaltError as e:
            logger.error("Gap Detection Agent halted (permanent error): %s", e)
            duration_ms = int((time.time() - start_time) * 1000)
            self._log_agent_action(
                session_id, "GapDetectionAgent",
                f"Items: {len(extraction_result.items)}",
                f"Permanent error: {str(e.original_error)[:200]}",
                duration_ms,
            )
            errors.append({"step": "gap_detection", "error": str(e.original_error)})
            session_state.status = "permanent_failure"
            session_state.errors = errors
            self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())
            return SessionResult(
                session_id=session_id,
                status="permanent_failure",
                session_state=session_state,
                errors=errors,
            )
        except RetryExhaustedError as e:
            logger.error("Gap Detection Agent retries exhausted: %s", e)
            duration_ms = int((time.time() - start_time) * 1000)
            self._log_agent_action(
                session_id, "GapDetectionAgent",
                f"Items: {len(extraction_result.items)}",
                f"Retries exhausted ({e.attempts} attempts): {str(e.original_error)[:200]}",
                duration_ms,
            )
            errors.append({"step": "gap_detection", "error": str(e.original_error)})
            session_state.status = "partial_failure"
            session_state.errors = errors
            self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())
            return SessionResult(
                session_id=session_id,
                status="partial_failure",
                session_state=session_state,
                errors=errors,
            )
        except Exception as e:
            logger.error("Gap Detection Agent failed: %s", e)
            duration_ms = int((time.time() - start_time) * 1000)
            self._log_agent_action(
                session_id, "GapDetectionAgent",
                f"Items: {len(extraction_result.items)}",
                f"Error: {str(e)[:200]}",
                duration_ms,
            )
            errors.append({"step": "gap_detection", "error": str(e)})
            session_state.status = "partial_failure"
            session_state.errors = errors
            self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())
            return SessionResult(
                session_id=session_id,
                status="partial_failure",
                session_state=session_state,
                errors=errors,
            )

        duration_ms = int((time.time() - start_time) * 1000)
        self._log_agent_action(
            session_id, "GapDetectionAgent",
            f"Items: {len(extraction_result.items)}",
            f"New: {gap_report.total_new}, Dup: {gap_report.total_duplicates}, Conflict: {gap_report.total_conflicts}",
            duration_ms,
        )

        # Store gap report (Requirement 2.4)
        session_state.gap_report = gap_report
        self._memory.store_intermediate(session_id, "gap_report", gap_report.model_dump())

        # Now store extracted items in Long-Term Memory for future sessions
        if extraction_result.items:
            search_items = [
                {
                    "item_id": f"{session_id}_item_{i}",
                    "content": item.text,
                    "item_type": item.item_type,
                    "confidence": item.confidence,
                }
                for i, item in enumerate(extraction_result.items)
            ]
            self._memory.store_for_search(session_id, search_items)

        # Step 3: Invoke Story Writer Agent (Requirement 2.3)
        start_time = time.time()
        try:
            stories = await self._invoke_with_retry(
                self._story_writer.generate_stories, gap_report.entries
            )
            epics = self._story_writer.group_into_epics(stories)
            serialization_result = self._story_writer.serialize_output(epics, session_id)
        except PipelineHaltError as e:
            logger.error("Story Writer Agent halted (permanent error): %s", e)
            duration_ms = int((time.time() - start_time) * 1000)
            self._log_agent_action(
                session_id, "StoryWriterAgent",
                f"Entries: {len(gap_report.entries)}",
                f"Permanent error: {str(e.original_error)[:200]}",
                duration_ms,
            )
            errors.append({"step": "story_writer", "error": str(e.original_error)})
            session_state.status = "permanent_failure"
            session_state.errors = errors
            self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())
            return SessionResult(
                session_id=session_id,
                status="permanent_failure",
                session_state=session_state,
                errors=errors,
            )
        except RetryExhaustedError as e:
            logger.error("Story Writer Agent retries exhausted: %s", e)
            duration_ms = int((time.time() - start_time) * 1000)
            self._log_agent_action(
                session_id, "StoryWriterAgent",
                f"Entries: {len(gap_report.entries)}",
                f"Retries exhausted ({e.attempts} attempts): {str(e.original_error)[:200]}",
                duration_ms,
            )
            errors.append({"step": "story_writer", "error": str(e.original_error)})
            session_state.status = "partial_failure"
            session_state.errors = errors
            self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())
            return SessionResult(
                session_id=session_id,
                status="partial_failure",
                session_state=session_state,
                errors=errors,
            )
        except Exception as e:
            logger.error("Story Writer Agent failed: %s", e)
            duration_ms = int((time.time() - start_time) * 1000)
            self._log_agent_action(
                session_id, "StoryWriterAgent",
                f"Entries: {len(gap_report.entries)}",
                f"Error: {str(e)[:200]}",
                duration_ms,
            )
            errors.append({"step": "story_writer", "error": str(e)})
            session_state.status = "partial_failure"
            session_state.errors = errors
            self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())
            return SessionResult(
                session_id=session_id,
                status="partial_failure",
                session_state=session_state,
                errors=errors,
            )

        duration_ms = int((time.time() - start_time) * 1000)

        story_output = serialization_result.output
        if serialization_result.errors:
            error_desc = f"{len(serialization_result.errors)} serialization errors"
            for ser_err in serialization_result.errors:
                errors.append({
                    "step": "story_writer_serialization",
                    "item_title": ser_err.item_title,
                    "failing_field": ser_err.failing_field,
                    "description": ser_err.description,
                })
        else:
            error_desc = ""

        output_summary = f"Stories: {len(stories)}, Epics: {len(epics)}"
        if error_desc:
            output_summary += f", {error_desc}"

        self._log_agent_action(
            session_id, "StoryWriterAgent",
            f"Entries: {len(gap_report.entries)}",
            output_summary,
            duration_ms,
        )

        # Store final output (Requirement 2.4)
        session_state.story_output = story_output
        if story_output:
            self._memory.store_intermediate(session_id, "story_output", story_output.model_dump())

        # Determine final status
        if errors:
            session_state.status = "partial_failure"
        else:
            session_state.status = "completed"

        session_state.errors = errors
        self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())

        return SessionResult(
            session_id=session_id,
            status=session_state.status,
            output=story_output,
            session_state=session_state,
            errors=errors,
        )

    def _validate_backlog_tickets(
        self,
        session_id: str,
        tickets: list[BacklogTicket],
        errors: list[dict],
    ) -> list[BacklogTicket]:
        """Validate backlog tickets against the BacklogTicket schema.

        Tickets that are already BacklogTicket instances (validated by Pydantic
        at SessionInputs construction time) are considered valid. This method
        provides an additional validation pass and logs any issues.

        For raw dict-based tickets that need validation, this method attempts
        Pydantic validation and collects errors for invalid entries.

        Args:
            session_id: The session identifier for logging.
            tickets: List of backlog tickets to validate.
            errors: Mutable list to append validation errors to.

        Returns:
            List of valid BacklogTicket instances.

        Requirements: 1.4, 1.6
        """
        valid_tickets: list[BacklogTicket] = []

        for i, ticket in enumerate(tickets):
            try:
                # Re-validate through Pydantic to catch any issues
                validated = BacklogTicket.model_validate(ticket.model_dump())
                valid_tickets.append(validated)
            except ValidationError as e:
                logger.warning(
                    "Backlog ticket %d failed validation: %s", i, e
                )
                errors.append({
                    "step": "backlog_validation",
                    "ticket_index": i,
                    "error": str(e),
                })
                self._log_agent_action(
                    session_id, "OrchestratorAgent",
                    f"Validating backlog ticket index {i}",
                    f"Validation failed: {str(e)[:200]}",
                    0,
                )

        logger.info(
            "Backlog ticket validation: %d valid, %d rejected out of %d total",
            len(valid_tickets),
            len(tickets) - len(valid_tickets),
            len(tickets),
        )

        return valid_tickets

    def validate_backlog_tickets_from_json(
        self,
        session_id: str,
        raw_tickets: list[dict],
    ) -> tuple[list[BacklogTicket], list[dict]]:
        """Validate raw JSON/dict backlog ticket data against the BacklogTicket schema.

        Accepts raw dictionaries (e.g., parsed from a JSON file) and validates each
        entry against the BacklogTicket Pydantic model. Valid entries are returned as
        BacklogTicket instances. Invalid entries are rejected with logged validation
        errors and processing continues with valid tickets only.

        This method is the primary entry point for validating raw JSON data that has
        not yet been parsed into BacklogTicket instances.

        Args:
            session_id: The session identifier for audit logging.
            raw_tickets: List of raw dictionaries to validate against BacklogTicket schema.

        Returns:
            A tuple of (valid_tickets, validation_errors) where:
            - valid_tickets: List of successfully validated BacklogTicket instances.
            - validation_errors: List of error dicts for rejected entries, each containing
              'step', 'ticket_index', 'raw_data', and 'error' keys.

        Requirements: 1.4, 1.6
        """
        valid_tickets: list[BacklogTicket] = []
        validation_errors: list[dict] = []

        for i, raw_ticket in enumerate(raw_tickets):
            try:
                validated = BacklogTicket.model_validate(raw_ticket)
                valid_tickets.append(validated)
            except ValidationError as e:
                logger.warning(
                    "Raw backlog ticket at index %d failed schema validation: %s",
                    i, e,
                )
                error_entry = {
                    "step": "backlog_validation",
                    "ticket_index": i,
                    "raw_data": raw_ticket,
                    "error": str(e),
                }
                validation_errors.append(error_entry)
                self._log_agent_action(
                    session_id, "OrchestratorAgent",
                    f"Validating raw backlog ticket index {i}",
                    f"Schema validation failed: {str(e)[:200]}",
                    0,
                )
            except Exception as e:
                # Handle non-validation errors (e.g., if raw_ticket is not a dict)
                logger.warning(
                    "Raw backlog ticket at index %d caused unexpected error: %s",
                    i, e,
                )
                error_entry = {
                    "step": "backlog_validation",
                    "ticket_index": i,
                    "raw_data": raw_ticket,
                    "error": f"Unexpected error: {str(e)}",
                }
                validation_errors.append(error_entry)
                self._log_agent_action(
                    session_id, "OrchestratorAgent",
                    f"Validating raw backlog ticket index {i}",
                    f"Unexpected error: {str(e)[:200]}",
                    0,
                )

        logger.info(
            "Raw backlog ticket validation: %d valid, %d rejected out of %d total",
            len(valid_tickets),
            len(validation_errors),
            len(raw_tickets),
        )

        return valid_tickets, validation_errors

    def _log_agent_action(
        self,
        session_id: str,
        agent_name: str,
        input_summary: str,
        output_summary: str,
        duration_ms: int,
    ) -> None:
        """Log an agent action to the audit log.

        Args:
            session_id: The session identifier.
            agent_name: Name of the agent that performed the action.
            input_summary: Summary of the input (max 500 chars).
            output_summary: Summary of the output (max 500 chars).
            duration_ms: Duration of the action in milliseconds.
        """
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc),
            agent_name=agent_name,
            input_summary=input_summary[:500],
            output_summary=output_summary[:500],
            duration_ms=duration_ms,
        )
        self._memory.log_action(session_id, entry)

    async def _invoke_with_retry(
        self,
        coro_func: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> T:
        """Invoke a coroutine with retry logic and timeout enforcement.

        Wraps the coroutine call in asyncio.wait_for with the specified timeout.
        Retries on transient errors (TransientToolError, asyncio.TimeoutError)
        with exponential backoff (1s, 2s, 4s). Halts immediately on permanent
        errors (PermanentToolError).

        Args:
            coro_func: An async callable to invoke.
            *args: Positional arguments to pass to the callable.
            max_retries: Maximum number of retries (default 3, so 4 total attempts).
            timeout: Timeout in seconds per invocation attempt (default 120).

        Returns:
            The result of the coroutine function call.

        Raises:
            PipelineHaltError: If a permanent error is encountered (no retry).
            RetryExhaustedError: If all retry attempts are exhausted.

        Requirements: 2.5, 2.6, 7.1, 7.3, 7.5, 7.6
        """
        backoff = 1.0
        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    coro_func(*args),
                    timeout=timeout,
                )
                return result
            except PermanentToolError as e:
                raise PipelineHaltError(e) from e
            except (TransientToolError, asyncio.TimeoutError) as e:
                if attempt == max_retries:
                    raise RetryExhaustedError(e, attempts=max_retries + 1) from e
                await asyncio.sleep(backoff)
                backoff *= 2
