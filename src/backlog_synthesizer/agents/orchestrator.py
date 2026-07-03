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
from backlog_synthesizer.agents.react_reasoning import ReActReasoner
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
from backlog_synthesizer.models.output import Epic, StoryOutput
from backlog_synthesizer.observability.tracing import PipelineTracer
from backlog_synthesizer.observability.metrics import PipelineMetrics
from backlog_synthesizer.tools.errors import PermanentToolError, TransientToolError
from backlog_synthesizer.tools.interfaces import LLMGenerationTool

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
        metadata: Additional metadata from ReAct decisions or other sources.
    """

    def __init__(
        self,
        session_id: str,
        status: str,
        output: StoryOutput | None = None,
        session_state: SessionState | None = None,
        errors: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.session_id = session_id
        self.status = status
        self.output = output
        self.session_state = session_state
        self.errors = errors or []
        self.metadata = metadata or {}


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
        reasoning_llm: LLMGenerationTool | None = None,
    ) -> None:
        """Initialize OrchestratorAgent with sub-agents and memory engine.

        Args:
            parser: The Parser Agent for document ingestion and extraction.
            gap_detector: The Gap Detection Agent for duplicate/conflict analysis.
            story_writer: The Story Writer Agent for user story generation.
            memory: The Memory Engine for session state and persistence.
            reasoning_llm: Optional LLM tool for ReAct reasoning at decision points.
                          When None, the orchestrator runs as a pure sequential pipeline.
        """
        self._parser = parser
        self._gap_detector = gap_detector
        self._story_writer = story_writer
        self._memory = memory
        self._reasoner = ReActReasoner(reasoning_llm) if reasoning_llm else None

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
        pipeline_start = time.time()

        # Initialize observability
        pipeline_tracer = PipelineTracer(
            session_id=session_id,
            attributes={
                "document_count": len(inputs.documents),
                "backlog_ticket_count": len(inputs.backlog_tickets),
            },
        )
        pipeline_metrics = PipelineMetrics()

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
                    "tags": ",".join(ticket.tags) if ticket.tags else "",
                    "created_at": ticket.created_at.isoformat() if ticket.created_at else "",
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
        logger.info(
            "ParserAgent completed: %d documents → %d items in %dms",
            len(non_backlog_docs), len(extraction_result.items), duration_ms,
        )
        self._log_agent_action(
            session_id, "ParserAgent",
            f"Documents: {len(non_backlog_docs)}",
            f"Extracted {len(extraction_result.items)} items, {len(extraction_result.errors)} errors",
            duration_ms,
        )

        # Store extraction results (Requirement 2.4)
        session_state.extraction_result = extraction_result
        self._memory.store_intermediate(session_id, "extraction_result", extraction_result.model_dump())

        # Record parser observability
        pipeline_tracer.record_agent_span("agent.parser", {
            "documents_processed": len(non_backlog_docs),
            "items_extracted": len(extraction_result.items),
            "errors": len(extraction_result.errors),
        }, duration_ms)
        pipeline_metrics.record_latency("parser", duration_ms)

        # --- Decision Point 1: After Parser — empty or few items ---
        if self._reasoner is not None:
            dp1_triggered = False
            try:
                if len(extraction_result.items) == 0:
                    dp1_triggered = True
                    decision = await self._reasoner.decide(
                        decision_point="after_parser_empty",
                        observation=f"Parser extracted 0 items from {len(non_backlog_docs)} documents",
                        available_actions=[
                            {"action": "halt", "description": "Stop pipeline — no actionable items found"},
                            {"action": "proceed_empty", "description": "Continue with empty result — gap detection will classify as all-new"},
                        ],
                    )
                    self._log_agent_action(
                        session_id, "ReActReasoner",
                        f"Decision point: after_parser_empty | Observation: Parser extracted 0 items from {len(non_backlog_docs)} documents",
                        f"Action: {decision['action']} | Reason: {decision['reason']}",
                        0,
                    )
                    if decision["action"] == "halt":
                        session_state.status = "completed"
                        session_state.errors = errors
                        self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())
                        return SessionResult(
                            session_id=session_id,
                            status="completed",
                            output=None,
                            session_state=session_state,
                            errors=errors,
                            metadata={"react_note": "Halted: no actionable items found after parsing"},
                        )
                elif len(extraction_result.items) < 3:
                    dp1_triggered = True
                    decision = await self._reasoner.decide(
                        decision_point="after_parser_few_items",
                        observation=f"Parser extracted only {len(extraction_result.items)} items (low count)",
                        available_actions=[
                            {"action": "proceed_with_warning", "description": "Continue but add low-confidence warning to output"},
                            {"action": "proceed_normal", "description": "Continue normally"},
                        ],
                    )
                    self._log_agent_action(
                        session_id, "ReActReasoner",
                        f"Decision point: after_parser_few_items | Observation: Parser extracted only {len(extraction_result.items)} items (low count)",
                        f"Action: {decision['action']} | Reason: {decision['reason']}",
                        0,
                    )
                    if decision["action"] == "proceed_with_warning":
                        errors.append({"step": "react_warning", "message": "Low item count — results may be incomplete"})
            except Exception as e:
                logger.warning("ReAct decision point 1 failed: %s. Continuing normally.", e)
            if not dp1_triggered:
                self._log_agent_action(
                    session_id, "ReActReasoner",
                    "Decision point: after_parser | Condition: normal (items >= 3, no errors)",
                    "Skipped: no abnormal condition detected",
                    0,
                )

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

            # --- Decision Point 4: On permanent error ---
            if self._reasoner is not None:
                try:
                    decision = await self._reasoner.decide(
                        decision_point="permanent_error",
                        observation=f"Permanent error in gap_detection: {str(e.original_error)[:200]}",
                        available_actions=[
                            {"action": "return_partial", "description": "Return whatever was completed before the error"},
                            {"action": "halt_completely", "description": "Fail the entire session"},
                        ],
                    )
                    self._log_agent_action(
                        session_id, "ReActReasoner",
                        f"Decision point: permanent_error | Observation: Permanent error in gap_detection: {str(e.original_error)[:100]}",
                        f"Action: {decision['action']} | Reason: {decision['reason']}",
                        0,
                    )
                    if decision["action"] == "return_partial" and extraction_result is not None:
                        session_state.status = "partial_failure"
                        session_state.errors = errors
                        self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())
                        return SessionResult(
                            session_id=session_id,
                            status="partial_failure",
                            session_state=session_state,
                            errors=errors,
                        )
                except Exception as react_err:
                    logger.warning("ReAct decision point 4 failed: %s. Halting completely.", react_err)

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
        logger.info(
            "GapDetectionAgent completed: %d items → new=%d, dup=%d, conflict=%d in %dms",
            len(extraction_result.items), gap_report.total_new,
            gap_report.total_duplicates, gap_report.total_conflicts, duration_ms,
        )
        self._log_agent_action(
            session_id, "GapDetectionAgent",
            f"Items: {len(extraction_result.items)}",
            f"New: {gap_report.total_new}, Dup: {gap_report.total_duplicates}, Conflict: {gap_report.total_conflicts}",
            duration_ms,
        )

        # Store gap report (Requirement 2.4)
        session_state.gap_report = gap_report
        self._memory.store_intermediate(session_id, "gap_report", gap_report.model_dump())

        # Record gap detection observability
        pipeline_tracer.record_agent_span("agent.gap_detection", {
            "items_processed": len(extraction_result.items),
            "duplicate_count": gap_report.total_duplicates,
            "conflict_count": gap_report.total_conflicts,
            "new_count": gap_report.total_new,
            "unprocessed_count": gap_report.total_unprocessed,
        }, duration_ms)
        pipeline_metrics.record_latency("gap_detection", duration_ms)
        pipeline_metrics.record_items("duplicate", gap_report.total_duplicates)
        pipeline_metrics.record_items("conflict", gap_report.total_conflicts)
        pipeline_metrics.record_items("new", gap_report.total_new)

        # --- Decision Point 2: After Gap Detection — all or mostly duplicates ---
        if self._reasoner is not None:
            dp2_triggered = False
            try:
                if gap_report.total_duplicates == len(gap_report.entries) and len(gap_report.entries) > 0:
                    dp2_triggered = True
                    decision = await self._reasoner.decide(
                        decision_point="after_gap_all_duplicates",
                        observation=f"All {len(gap_report.entries)} items are duplicates of existing backlog tickets",
                        available_actions=[
                            {"action": "halt_all_duplicates", "description": "Stop — all items already in backlog"},
                            {"action": "proceed_anyway", "description": "Continue to generate stories even for duplicates"},
                        ],
                    )
                    self._log_agent_action(
                        session_id, "ReActReasoner",
                        f"Decision point: after_gap_all_duplicates | Observation: All {len(gap_report.entries)} items are duplicates",
                        f"Action: {decision['action']} | Reason: {decision['reason']}",
                        0,
                    )
                    if decision["action"] == "halt_all_duplicates":
                        session_state.status = "completed"
                        session_state.errors = errors
                        self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())
                        return SessionResult(
                            session_id=session_id,
                            status="completed",
                            output=None,
                            session_state=session_state,
                            errors=errors,
                            metadata={"react_note": "Halted: all items are duplicates of existing backlog"},
                        )
                elif len(gap_report.entries) > 0 and gap_report.total_duplicates / max(len(gap_report.entries), 1) > 0.8:
                    dp2_triggered = True
                    dup_pct = int(gap_report.total_duplicates / len(gap_report.entries) * 100)
                    decision = await self._reasoner.decide(
                        decision_point="after_gap_mostly_duplicates",
                        observation=f"{dup_pct}% of items are duplicates",
                        available_actions=[
                            {"action": "proceed_with_warning", "description": "Continue but add high-duplicate warning to output"},
                            {"action": "proceed_normal", "description": "Continue normally"},
                        ],
                    )
                    self._log_agent_action(
                        session_id, "ReActReasoner",
                        f"Decision point: after_gap_mostly_duplicates | Observation: {dup_pct}% of items are duplicates",
                        f"Action: {decision['action']} | Reason: {decision['reason']}",
                        0,
                    )
                    if decision["action"] == "proceed_with_warning":
                        errors.append({"step": "react_warning", "message": f"High duplicate rate ({dup_pct}%) — most items already in backlog"})
            except Exception as e:
                logger.warning("ReAct decision point 2 failed: %s. Continuing normally.", e)
            if not dp2_triggered:
                self._log_agent_action(
                    session_id, "ReActReasoner",
                    "Decision point: after_gap_detection | Condition: normal (mixed classifications)",
                    "Skipped: no abnormal condition detected",
                    0,
                )

        # --- Decision Point 5: Conflicts detected ---
        conflict_summary_requested = False
        conflict_rate = gap_report.total_conflicts / max(len(gap_report.entries), 1)
        if self._reasoner is not None and gap_report.total_conflicts > 0 and conflict_rate > 0.2:
            try:
                decision = await self._reasoner.decide(
                    decision_point="conflicts_detected",
                    observation=f"{gap_report.total_conflicts} conflicts detected between new items and existing backlog",
                    available_actions=[
                        {"action": "proceed_with_conflicts", "description": "Pass conflicts to Story Writer normally"},
                        {"action": "add_conflict_summary", "description": "Generate stories and add a conflict summary to metadata"},
                    ],
                )
                self._log_agent_action(
                    session_id, "ReActReasoner",
                    f"Decision point: conflicts_detected | Observation: {gap_report.total_conflicts} conflicts detected",
                    f"Action: {decision['action']} | Reason: {decision['reason']}",
                    0,
                )
                if decision["action"] == "add_conflict_summary":
                    conflict_summary_requested = True
            except Exception as e:
                logger.warning("ReAct decision point 5 failed: %s. Continuing normally.", e)
        elif self._reasoner is not None and gap_report.total_conflicts > 0:
            # Conflicts exist but below 20% threshold — skip reasoning
            self._log_agent_action(
                session_id, "ReActReasoner",
                f"Decision point: conflicts_detected | Condition: {gap_report.total_conflicts} conflicts ({conflict_rate*100:.0f}%), below 20% threshold",
                "Skipped: conflict rate below threshold",
                0,
            )

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

            # --- Decision Point 4: On permanent error ---
            if self._reasoner is not None:
                try:
                    decision = await self._reasoner.decide(
                        decision_point="permanent_error",
                        observation=f"Permanent error in story_writer: {str(e.original_error)[:200]}",
                        available_actions=[
                            {"action": "return_partial", "description": "Return whatever was completed before the error"},
                            {"action": "halt_completely", "description": "Fail the entire session"},
                        ],
                    )
                    self._log_agent_action(
                        session_id, "ReActReasoner",
                        f"Decision point: permanent_error | Observation: Permanent error in story_writer: {str(e.original_error)[:100]}",
                        f"Action: {decision['action']} | Reason: {decision['reason']}",
                        0,
                    )
                    if decision["action"] == "return_partial" and gap_report is not None:
                        session_state.status = "partial_failure"
                        session_state.errors = errors
                        self._memory.store_intermediate(session_id, "session_state", session_state.model_dump())
                        return SessionResult(
                            session_id=session_id,
                            status="partial_failure",
                            session_state=session_state,
                            errors=errors,
                        )
                except Exception as react_err:
                    logger.warning("ReAct decision point 4 failed: %s. Halting completely.", react_err)

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

        # Record story writer observability
        pipeline_tracer.record_agent_span("agent.story_writer", {
            "items_received": len(gap_report.entries),
            "stories_generated": len(stories),
            "epic_count": len(epics),
        }, duration_ms)
        pipeline_metrics.record_latency("story_writer", duration_ms)

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
        logger.info(
            "StoryWriterAgent completed: %d entries → %d stories, %d epics in %dms",
            len(gap_report.entries), len(stories), len(epics), duration_ms,
        )

        # --- Decision Point 3: After Story Writer — quality issues ---
        if self._reasoner is not None:
            dp3_triggered = False
            try:
                refinement_count = sum(1 for s in stories if s.needs_refinement)
                if refinement_count > len(stories) / 2 and len(stories) > 0:
                    dp3_triggered = True
                    decision = await self._reasoner.decide(
                        decision_point="after_story_writer_quality",
                        observation=f"{refinement_count} of {len(stories)} stories need refinement",
                        available_actions=[
                            {"action": "return_with_warning", "description": "Return stories as-is with quality warning"},
                            {"action": "return_normal", "description": "Return without special warning"},
                        ],
                    )
                    self._log_agent_action(
                        session_id, "ReActReasoner",
                        f"Decision point: after_story_writer_quality | Observation: {refinement_count} of {len(stories)} stories need refinement",
                        f"Action: {decision['action']} | Reason: {decision['reason']}",
                        0,
                    )
                    if decision["action"] == "return_with_warning":
                        errors.append({"step": "react_warning", "message": f"Quality concern: {refinement_count}/{len(stories)} stories need refinement"})

                if len(epics) == 0 and len(stories) > 0:
                    dp3_triggered = True
                    decision = await self._reasoner.decide(
                        decision_point="after_story_writer_no_epics",
                        observation="Stories generated but no epics formed (no shared tags)",
                        available_actions=[
                            {"action": "return_ungrouped", "description": "Return stories without epic grouping"},
                            {"action": "return_single_epic", "description": "Group all stories under a single generic epic"},
                        ],
                    )
                    self._log_agent_action(
                        session_id, "ReActReasoner",
                        f"Decision point: after_story_writer_no_epics | Observation: Stories generated but no epics formed",
                        f"Action: {decision['action']} | Reason: {decision['reason']}",
                        0,
                    )
                    if decision["action"] == "return_single_epic":
                        epics = [Epic(epic_title="General", stories=list(stories))]
                        serialization_result = self._story_writer.serialize_output(epics, session_id)
                        story_output = serialization_result.output
            except Exception as e:
                logger.warning("ReAct decision point 3 failed: %s. Continuing normally.", e)
            if not dp3_triggered:
                self._log_agent_action(
                    session_id, "ReActReasoner",
                    "Decision point: after_story_writer | Condition: normal (quality acceptable)",
                    "Skipped: no abnormal condition detected",
                    0,
                )

        # --- Decision Point 5 (continued): Add conflict summary to metadata if requested ---
        if conflict_summary_requested and story_output is not None:
            conflict_entries = [
                entry for entry in gap_report.entries
                if entry.classification == "conflict"
            ]
            conflict_details = [
                {
                    "item_text": entry.item.text[:100],
                    "similar_ticket_id": entry.similar_ticket_id,
                    "similarity_score": entry.similarity_score,
                }
                for entry in conflict_entries
            ]
            errors.append({
                "step": "react_conflict_summary",
                "message": f"{len(conflict_details)} conflicts detected",
                "conflicts": conflict_details,
            })

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

        pipeline_total_ms = int((time.time() - pipeline_start) * 1000)
        logger.info(
            "Pipeline completed: session=%s, status=%s, items=%d, stories=%d, "
            "epics=%d, total_time=%dms",
            session_id,
            session_state.status,
            len(extraction_result.items) if extraction_result else 0,
            len(stories),
            len(epics),
            pipeline_total_ms,
        )

        # Save session replay for observability
        pipeline_tracer.save_session_replay(session_state.status, pipeline_metrics.to_dict())

        return SessionResult(
            session_id=session_id,
            status=session_state.status,
            output=story_output,
            session_state=session_state,
            errors=errors,
        )

        # --- End of pipeline --- timing is logged above per agent

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
