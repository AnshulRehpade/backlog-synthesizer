# Implementation Plan: Backlog Synthesizer

## Overview

This plan implements a multi-agent system in Python using an orchestrator pattern with specialized sub-agents (Parser, Gap Detection, Story Writer) backed by a Memory Engine. The implementation follows a bottom-up approach: data models and tool interfaces first, then individual agents, then orchestration and wiring, and finally the evaluation framework.

## Tasks

- [x] 1. Set up project structure, data models, and tool interfaces
  - [x] 1.1 Create project directory structure and configuration files
    - Create `src/backlog_synthesizer/` package with `__init__.py`
    - Create sub-packages: `models/`, `agents/`, `tools/`, `memory/`, `evaluation/`
    - Create `pyproject.toml` with dependencies (pydantic, chromadb, hypothesis, pytest, asyncio)
    - Create `tests/` directory with `conftest.py`
    - _Requirements: 9.1, 9.2, 9.3, 9.7_

  - [x] 1.2 Implement all Pydantic data models
    - Create `src/backlog_synthesizer/models/inputs.py` with `DocumentType`, `InputDocument`, `BacklogTicket`, `SessionInputs`
    - Create `src/backlog_synthesizer/models/extraction.py` with `TextChunk`, `ExtractedItem`, `ExtractionResult`, `DocumentError`
    - Create `src/backlog_synthesizer/models/gap_detection.py` with `DuplicateFlag`, `ConflictFlag`, `GapReportEntry`, `GapReport`
    - Create `src/backlog_synthesizer/models/output.py` with `AcceptanceCriterion`, `UserStory`, `Epic`, `OutputMetadata`, `StoryOutput`
    - Create `src/backlog_synthesizer/models/memory.py` with `AuditEntry`, `SessionState`
    - Create `src/backlog_synthesizer/models/evaluation.py` with `GoldenEntry`, `JudgeScores`, `EvaluationCaseResult`, `EvaluationReport`
    - _Requirements: 1.4, 1.5, 3.1, 3.2, 3.3, 3.4, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5, 6.4, 10.1, 10.4_

  - [x] 1.3 Write property tests for data model validation
    - **Property 1: Output JSON round-trip serialization** — verify `StoryOutput.model_validate_json(output.model_dump_json()) == output`
    - **Validates: Requirements 10.2**
    - **Property 4: Extracted item structural validity** — verify non-empty text, non-negative chunk index, confidence in [0.0, 1.0]
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    - **Property 7: User story structural validity** — verify pattern match, 2-10 acceptance criteria, 1-5 tags
    - **Validates: Requirements 5.1, 5.2, 5.3**
    - **Property 16: Output index array consistency** — verify index matches epics count and titles
    - **Validates: Requirements 10.3**

  - [x] 1.4 Define tool interface protocols and error types
    - Create `src/backlog_synthesizer/tools/interfaces.py` with Protocol classes: `DocumentParsingTool`, `EmbeddingTool`, `VectorSearchTool`, `LLMGenerationTool`
    - Create `src/backlog_synthesizer/tools/errors.py` with `ToolError`, `TransientToolError`, `PermanentToolError`
    - Define typed method signatures with input/return/error types for each protocol
    - _Requirements: 9.1, 9.2, 9.3, 9.5, 9.6_

  - [x] 1.5 Write property test for tool error translation
    - **Property 15: Tool error translation** — verify implementation-specific exceptions are translated to interface-defined error types
    - **Validates: Requirements 9.6**

- [x] 2. Implement Memory Engine
  - [x] 2.1 Implement Short-Term Memory with fallback
    - Create `src/backlog_synthesizer/memory/short_term.py`
    - Implement in-session state store accessible by session ID
    - Implement fallback to Python dict when primary store is unavailable
    - Log warning to Audit Log when using fallback
    - _Requirements: 6.1, 6.6_

  - [x] 2.2 Implement Long-Term Memory with Chroma vector store
    - Create `src/backlog_synthesizer/memory/long_term.py`
    - Implement embedding generation and storage via EmbeddingTool interface
    - Implement semantic search via VectorSearchTool interface
    - Configure 30-day retention policy for stored entries
    - _Requirements: 6.2, 6.3, 6.5_

  - [x] 2.3 Implement Audit Log
    - Create `src/backlog_synthesizer/memory/audit_log.py`
    - Record sub-agent invocations with timestamp, agent name, input/output summaries (max 500 chars), duration
    - Implement retrieval by session ID in chronological order
    - Return empty result for invalid/expired session IDs with appropriate message
    - _Requirements: 6.4, 6.7, 6.8_

  - [x] 2.4 Implement Memory Engine facade
    - Create `src/backlog_synthesizer/memory/engine.py` with `MemoryEngine` class
    - Wire Short-Term Memory, Long-Term Memory, and Audit Log together
    - Implement `store_intermediate`, `store_for_search`, `log_action`, `get_audit_log` methods
    - _Requirements: 6.1, 6.2, 6.4, 6.7_

  - [x] 2.5 Write property tests for Memory Engine
    - **Property 13: Audit log chronological ordering** — verify entries returned sorted by timestamp ascending
    - **Validates: Requirements 6.7**
    - **Property 14: Short-term memory round-trip by session ID** — verify store/retrieve returns equal payload
    - **Validates: Requirements 6.1**

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement Parser Agent
  - [x] 4.1 Implement document chunking logic
    - Create `src/backlog_synthesizer/agents/parser.py`
    - Implement `_chunk_text` method with max_tokens=2000 and overlap=200 tokens
    - Handle edge cases: empty text, text shorter than max_tokens
    - _Requirements: 3.5_

  - [x] 4.2 Write property test for document chunking
    - **Property 2: Document chunking preserves content with bounded size** — verify chunk size ≤ 2000 tokens, 200 token overlap, and lossless reconstruction
    - **Validates: Requirements 3.5**

  - [x] 4.3 Implement document ingestion (text, PDF, HTML)
    - Implement text/markdown ingestion (.txt, .md) via direct read
    - Implement PDF ingestion via DocumentParsingTool interface
    - Implement HTML wiki export ingestion with markup stripping and heading hierarchy preservation
    - Return DocumentError for malformed/unreadable files with filename, reason, byte_offset/line_number
    - _Requirements: 1.1, 1.2, 1.3, 1.5_

  - [x] 4.4 Write property tests for document parsing
    - **Property 17: Malformed document error structure** — verify error contains non-empty filename, non-empty reason
    - **Validates: Requirements 1.5**
    - **Property 18: HTML stripping preserves heading hierarchy** — verify no HTML tags in output, heading order preserved
    - **Validates: Requirements 1.3**

  - [x] 4.5 Implement information extraction via LLM
    - Implement extraction of decisions, pain points, and feature requests from meeting transcripts
    - Implement extraction of technical constraints and architectural decisions from architecture docs
    - Structure extracted items with confidence scores, source chunk index, character offset
    - Return empty extraction result with metadata note when no items are found
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6_

- [x] 5. Implement Gap Detection Agent
  - [x] 5.1 Implement semantic similarity comparison
    - Create `src/backlog_synthesizer/agents/gap_detection.py`
    - Compute embeddings for extracted items via EmbeddingTool interface
    - Query Long-Term Memory for similar backlog tickets via VectorSearchTool interface
    - Enforce 30-second timeout per request; mark as unprocessed on timeout/failure
    - _Requirements: 4.1, 4.6_

  - [x] 5.2 Implement duplicate, conflict, and new item classification
    - Apply threshold ≥ 0.85 → duplicate with matching ticket ID
    - Apply threshold 0.50–0.85 with contradicting statements → conflict with contradiction description
    - Apply threshold < 0.50 or no contradiction → new
    - Handle empty backlog case: mark all items as new with confidence 1.0
    - Produce gap report with counts of new, duplicates, conflicts, unprocessed
    - _Requirements: 4.2, 4.3, 4.4, 4.5_

  - [x] 5.3 Write property tests for gap detection
    - **Property 5: Gap classification correctness by similarity threshold** — verify classification logic based on thresholds
    - **Validates: Requirements 4.2, 4.3, 4.4**
    - **Property 6: Empty backlog marks all items as new** — verify all items classified as "new" with confidence 1.0
    - **Validates: Requirements 4.5**

- [x] 6. Implement Story Writer Agent
  - [x] 6.1 Implement user story generation
    - Create `src/backlog_synthesizer/agents/story_writer.py`
    - Generate stories via LLMGenerationTool with "As a [role], I want [goal], so that [benefit]" format
    - Generate 2-10 acceptance criteria per story
    - Assign 1-5 feature tags per story
    - Handle insufficient detail: produce story with placeholder text and "needs-refinement" tag
    - _Requirements: 5.1, 5.2, 5.3, 5.6_

  - [x] 6.2 Implement epic grouping logic
    - Group stories sharing at least one tag under a common Epic
    - Enforce epic title max 60 characters
    - _Requirements: 5.4_

  - [x] 6.3 Implement output serialization
    - Serialize output as JSON conforming to the StoryOutput schema
    - Generate top-level index array with epic titles and story counts
    - Handle serialization failures: return error object with item title, failing field, and description while including successfully serialized items
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 6.4 Write property tests for story generation and serialization
    - **Property 8: Epic grouping by shared tags** — verify stories with shared tags appear under a common Epic with title ≤ 60 chars
    - **Validates: Requirements 5.4**

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement Orchestrator Agent
  - [x] 8.1 Implement pipeline sequencing and session management
    - Create `src/backlog_synthesizer/agents/orchestrator.py`
    - Implement `run_session` method that invokes Parser → Gap Detection → Story Writer in order
    - Create session in Memory Engine at start, store intermediate results after each agent completes
    - Validate backlog ticket JSON schema; reject invalid entries, log errors, continue with valid ones
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 1.4, 1.6_

  - [x] 8.2 Implement retry logic and timeout enforcement
    - Implement `_invoke_with_retry` with exponential backoff (1s, 2s, 4s) for transient errors
    - Classify errors: transient (429, 500, 502, 503, 504, network timeout, sub-agent timeout) vs. permanent (401, 403, 404, auth failure, schema violation)
    - Enforce 120-second timeout per sub-agent invocation; treat timeout as transient
    - Halt immediately on permanent errors with "permanent_failure" status
    - Return partial results with "partial_failure" status when retries exhausted
    - _Requirements: 2.5, 2.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 8.3 Write property tests for orchestrator retry and error handling
    - **Property 9: Retry policy for transient errors** — verify up to 3 retries with 1s, 2s, 4s backoff
    - **Validates: Requirements 2.5, 7.1**
    - **Property 10: No retry for permanent errors** — verify immediate halt, no retry
    - **Validates: Requirements 2.6, 7.3**
    - **Property 11: Partial failure result structure** — verify status "partial_failure" with errors array
    - **Validates: Requirements 7.2**

- [x] 9. Implement concrete tool implementations and configuration
  - [x] 9.1 Implement concrete DocumentParsingTool
    - Create `src/backlog_synthesizer/tools/document_parsing.py`
    - Implement `pdf_to_text` using a PDF library (e.g., PyMuPDF)
    - Implement `chunk_text` with token-based splitting
    - Translate library-specific exceptions to `ToolError` subtypes
    - _Requirements: 9.1, 9.6_

  - [x] 9.2 Implement concrete EmbeddingTool and VectorSearchTool
    - Create `src/backlog_synthesizer/tools/embedding.py`
    - Create `src/backlog_synthesizer/tools/vector_search.py`
    - Implement embedding generation (e.g., via OpenAI or sentence-transformers)
    - Implement Chroma-backed vector search with store and query methods
    - Translate library-specific exceptions to `ToolError` subtypes
    - _Requirements: 9.2, 9.6_

  - [x] 9.3 Implement concrete LLMGenerationTool
    - Create `src/backlog_synthesizer/tools/llm_generation.py`
    - Implement `generate` method calling an LLM API (e.g., OpenAI, Anthropic)
    - Classify HTTP error codes into transient vs. permanent errors
    - Translate API-specific exceptions to `ToolError` subtypes
    - _Requirements: 9.3, 9.6_

  - [x] 9.4 Implement dependency injection configuration
    - Create `src/backlog_synthesizer/config.py` with `PipelineConfig` class
    - Implement configuration mechanism that binds concrete tool implementations to interfaces
    - Load configuration from environment variables or config file
    - Require zero changes to agent code when swapping tool implementations
    - _Requirements: 9.4, 9.7_

- [x] 10. Implement Evaluation Framework
  - [x] 10.1 Create golden dataset
    - Create `data/golden_dataset/` directory
    - Create at least 3 sample meeting transcripts, each containing at least one decision, one pain point, and one feature request
    - Create corresponding hand-written ideal UserStory outputs for each sample
    - _Requirements: 8.1_

  - [x] 10.2 Implement evaluation pipeline
    - Create `src/backlog_synthesizer/evaluation/framework.py`
    - Implement `run_evaluation` that executes full pipeline against each golden entry
    - Implement `_compute_keyword_overlap` as normalized case-insensitive token matching score
    - Implement `_llm_judge_score` scoring relevance, completeness, clarity on 1-5 scale
    - Produce JSON summary report with per-case and aggregate metrics (mean, minimum)
    - Record failure reason and score 0 for entries where pipeline fails; continue with remaining
    - _Requirements: 8.2, 8.3, 8.4, 8.5, 8.6_

  - [x] 10.3 Write property test for keyword overlap computation
    - **Property 12: Keyword overlap score computation** — verify normalized score in [0.0, 1.0], correct matching count, empty expected → 1.0
    - **Validates: Requirements 8.3**

- [x] 11. Integration wiring and backlog ticket validation
  - [x] 11.1 Implement backlog ticket schema validation in Orchestrator
    - Validate incoming JSON against BacklogTicket Pydantic model
    - Load valid tickets into Long-Term Memory
    - Reject invalid entries with logged validation errors
    - Continue processing with valid tickets only
    - _Requirements: 1.4, 1.6_

  - [x] 11.2 Write property test for backlog ticket validation
    - **Property 3: Backlog ticket validation accepts valid and rejects invalid** — verify accepted + rejected = total count
    - **Validates: Requirements 1.4, 1.6**

  - [x] 11.3 Wire all components together in main entry point
    - Create `src/backlog_synthesizer/main.py` as the application entry point
    - Instantiate PipelineConfig, load tool implementations from config
    - Instantiate MemoryEngine, all agents, and OrchestratorAgent
    - Expose `run` function that accepts file paths and returns serialized StoryOutput
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 9.7_

- [x] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All tool interactions go through Protocol interfaces for swappability
- Python with Pydantic provides JSON schema generation and validation out of the box

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.4"] },
    { "id": 2, "tasks": ["1.3", "1.5", "2.1", "2.2", "2.3"] },
    { "id": 3, "tasks": ["2.4"] },
    { "id": 4, "tasks": ["2.5", "4.1"] },
    { "id": 5, "tasks": ["4.2", "4.3"] },
    { "id": 6, "tasks": ["4.4", "4.5", "5.1"] },
    { "id": 7, "tasks": ["5.2"] },
    { "id": 8, "tasks": ["5.3", "6.1"] },
    { "id": 9, "tasks": ["6.2", "6.3"] },
    { "id": 10, "tasks": ["6.4", "8.1"] },
    { "id": 11, "tasks": ["8.2"] },
    { "id": 12, "tasks": ["8.3", "9.1", "9.2", "9.3"] },
    { "id": 13, "tasks": ["9.4", "10.1"] },
    { "id": 14, "tasks": ["10.2", "11.1"] },
    { "id": 15, "tasks": ["10.3", "11.2", "11.3"] }
  ]
}
```
