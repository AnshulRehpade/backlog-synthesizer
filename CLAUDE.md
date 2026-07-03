# CLAUDE.md — Backlog Synthesizer Project Context

This file is read at the start of every session to provide immediate architectural context. Do not re-discover from source — trust this document.

---

## 1. Project Summary

The Backlog Synthesizer is a multi-agent Python system that reads meeting transcripts, architecture documents, and existing backlog tickets, then produces structured user stories with acceptance criteria grouped into epics. It solves the problem of information loss between meetings and ticketing systems — decisions, pain points, and feature requests surface in conversations but never make it into actionable backlog items without manual effort. The output is a JSON document ready to import into JIRA, Linear, or GitHub Issues.

---

## 2. Architecture Overview

### Pipeline Flow

```
Input Files (.txt, .md, .pdf, .html, .json)
    ↓
Orchestrator Agent
    ↓ validates backlog tickets, loads into vector store
    ↓
Parser Agent (concurrent per document)
    ↓ chunks text → LLM extraction → structured items with tags
    ↓
Gap Detection Agent (concurrent per item)
    ↓ pre-filter by tag+status → embed → semantic search → classify
    ↓
Story Writer Agent (concurrent per item)
    ↓ LLM generation → user stories → epic grouping → JSON output
    ↓
StoryOutput JSON
```

### Agents

**Orchestrator Agent** (`agents/orchestrator.py`): Coordinates the sequential pipeline. Validates backlog tickets via Pydantic. Manages retry logic (3 retries, 1s/2s/4s exponential backoff for transient errors, immediate halt for permanent errors). Stores intermediate results in Memory Engine after each step. Enforces 120s timeout per sub-agent invocation.

**Parser Agent** (`agents/parser.py`): Ingests documents (text via decode, PDF via PyMuPDF, HTML with heading-hierarchy preservation). Chunks text into 2000-token segments with 200-token overlap. Sends chunks to LLM for extraction of decisions, pain points, feature requests, and constraints — each with confidence scores, source location, and 1-5 topic tags. Processes multiple documents concurrently via `asyncio.gather()`.

**Gap Detection Agent** (`agents/gap_detection.py`): Generates embeddings for extracted items via sentence-transformers. Pre-filters the vector store (exclude closed/archived tickets, optionally filter by matching tag). Runs semantic similarity search against the filtered set. Classifies each item: DUPLICATE (≥0.85), CONFLICT (0.50-0.85 with LLM-confirmed contradiction), or NEW (<0.50). Items that timeout or error become UNPROCESSED.

**Story Writer Agent** (`agents/story_writer.py`): Receives only NEW and CONFLICT items (duplicates skipped). Generates user stories via LLM with inherited tags from the parser. Groups stories into epics using union-find on shared tags (epic title ≤60 chars). Serializes output as JSON. Generates all stories concurrently via `asyncio.gather()`. Items with insufficient detail (<10 chars or confidence <0.3) produce placeholder stories tagged "needs-refinement".

### Memory Engine (`memory/engine.py`)

Three-tier system behind a facade:

- **Short-Term Memory** (`memory/short_term.py`): In-session state keyed by session ID. Python dict with optional primary store (e.g., Redis). Falls back to in-process dict with logged warning if primary unavailable.
- **Long-Term Memory** (`memory/long_term.py`): ChromaDB vector store for semantic search. Stores embeddings + metadata (status, tags, content, stored_at). 30-day retention policy. Used by Gap Detection for backlog comparison.
- **Audit Log** (`memory/audit_log.py`): Chronological record of every agent invocation. Each entry has timestamp, agent name, input/output summaries (max 500 chars), and duration_ms. Retrievable by session ID in ascending order.

### Tool Interfaces (`tools/interfaces.py`)

All external I/O is behind Python `Protocol` classes:

- `DocumentParsingTool`: pdf_to_text, chunk_text
- `EmbeddingTool`: generate_embedding
- `VectorSearchTool`: query_similar, query_similar_filtered, store
- `LLMGenerationTool`: generate(prompt, system_prompt)

Agents import only these protocols — never concrete implementations. Swapping providers (OpenAI ↔ Anthropic, Chroma ↔ FAISS) requires zero agent code changes.

### Concrete Implementations

- `tools/anthropic_generation.py` — Claude via Anthropic SDK (default)
- `tools/llm_generation.py` — OpenAI SDK (alternative)
- `tools/embedding.py` — sentence-transformers `all-MiniLM-L6-v2` (local, no API key)
- `tools/vector_search.py` — ChromaDB (ephemeral or persistent)
- `tools/document_parsing.py` — PyMuPDF for PDFs

---

## 3. Key Design Decisions

### Sequential pipeline (Parser → Gap Detection → Story Writer)

Story Writer needs Gap Detection's output to know which items are NEW vs DUPLICATE. Running them in parallel would mean Story Writer either processes duplicates (wasting LLM calls) or has no data. This ordering is a hard dependency, not a performance choice.

### Parallelism WITHIN agents, not between them

- Parser: multiple documents processed concurrently via `asyncio.gather()`
- Story Writer: multiple stories generated concurrently via `asyncio.gather()`
- Gap Detection: items processed concurrently via individual `asyncio.wait_for()` with timeout

### Three-level fallback in Gap Detection

1. **Tag + Status filter** → search only active tickets sharing the item's tag
2. **Status filter only** → search all non-closed tickets (if tag filter returned empty)
3. **Unfiltered** → search entire store (if no filter support or still empty)

This progressively widens the search until results are found, balancing precision vs recall.

### Union-find for epic grouping

Stories sharing tags are grouped transitively. If A shares a tag with B, and B shares a tag with C, all three land in the same epic. Simple, deterministic, but can produce overly broad epics when generic tags connect unrelated stories.

### Transient vs Permanent error taxonomy

- **Transient** (retry): HTTP 429, 500, 502, 503, 504, network timeout, sub-agent timeout → up to 3 retries with 1s, 2s, 4s backoff
- **Permanent** (halt): HTTP 401, 403, 404, auth failure, schema violation → immediate pipeline halt
- All tool errors are `ToolError → TransientToolError | PermanentToolError` — agents never see implementation-specific exceptions

### Configurable thresholds via environment

- `GAP_DETECTION_DUPLICATE_THRESHOLD` (default 0.85)
- `GAP_DETECTION_CONFLICT_THRESHOLD` (default 0.50)
- `LLM_PROVIDER` switches between anthropic/openai with zero code changes

---

## 4. Current Implementation State

### Fully built and working

- All 4 agents (Parser, Gap Detection, Story Writer, Orchestrator)
- **ReAct Orchestrator** — LLM-driven decision-making at 5 pipeline decision points
- Memory Engine (all 3 tiers)
- All tool implementations (Anthropic, OpenAI, embeddings, vector search, PDF parsing)
- Configuration system (env vars + optional JSON config file)
- Evaluation framework with golden dataset (3 transcripts + expected outputs)
- Pre-filtering in gap detection (tag + status)
- Tag extraction in parser, tag inheritance in story writer
- Concurrent execution within agents
- Timing instrumentation in orchestrator
- **tiktoken-based tokenizer** for accurate token counting in chunking (configurable via TOKENIZER_MODEL)
- **OpenTelemetry observability** — tracing (per-agent spans), metrics (token usage, latency, cost estimation), session replay (JSON in runs/)
- **Evaluation framework** — CLI with `run`/`compare` commands, 20 golden entries, regression detection, confidence intervals
- 442 tests passing (property-based + unit + integration)
- CI pipeline (GitHub Actions with coverage enforcement)
- CLI entry points (`demo.py` and `python -m backlog_synthesizer.main`)

### Partially implemented

- Persistent ChromaDB: supported via config but no cross-session deduplication workflow
- Prompt versioning: prompts exist but no formal version tracking system

### Not yet implemented

- Streaming output (currently blocks until entire pipeline completes)
- Web UI or API server
- Authentication/authorization for multi-tenant use

---

## 5. Conventions to Always Follow

1. **All external I/O behind Protocol interfaces** — never import a concrete tool class inside an agent module
2. **Error classification** — `ToolError` base class, `TransientToolError` for retryable, `PermanentToolError` for halt-worthy. Agents catch these, not implementation exceptions.
3. **Async throughout** — all agent entry methods (`parse_documents`, `analyze_gaps`, `generate_stories`, `run_session`) are `async`. Use `asyncio.to_thread()` to wrap synchronous tool calls.
4. **Pydantic models for all data contracts** — `ExtractedItem`, `GapReportEntry`, `UserStory`, `StoryOutput`, etc. No raw dicts crossing agent boundaries.
5. **Environment variables for all configuration** — loaded via `python-dotenv` in `main.py`. Config class in `config.py` reads them with sensible defaults.
6. **Tags flow through the pipeline** — Parser extracts them, Gap Detection filters by them, Story Writer inherits them for epic grouping.
7. **Tests mock all external tools** — no real API calls in tests. Use `MagicMock` for tools, `AsyncMock` for async agent methods.

---

## 6. What NOT to Do

- **Never run Story Writer before Gap Detection completes** — it needs the gap report to filter duplicates
- **Never add concrete tool dependencies inside agents** — agents depend only on Protocol interfaces defined in `tools/interfaces.py`
- **Never break the three-level fallback in Gap Detection** — tag+status → status → unfiltered. If you remove a level, large backlogs may return no results.
- **Never hardcode API keys or model names** — always use env vars via `PipelineConfig`
- **Never store extracted items in Long-Term Memory before Gap Detection runs** — they'd match against themselves as duplicates (this bug was already fixed)
- **Never use blocking I/O in async agent methods** — wrap with `asyncio.to_thread()`
- **Never pass raw dicts between agents** — use Pydantic models for type safety and validation

---

## 7. ReAct Orchestrator (Implemented)

The Orchestrator uses LLM-driven reasoning at 5 decision points within the sequential pipeline. The ReAct layer is purely advisory — if the LLM fails, the pipeline falls back to default behavior.

### Decision Points

1. **After Parser — empty/few items** (`react_reasoning.py` + orchestrator)
   - 0 items → LLM decides: halt or proceed_empty
   - <3 items → LLM decides: proceed_with_warning or proceed_normal
   - Default fallback: halt (empty) / proceed_with_warning (few)

2. **After Gap Detection — all/mostly duplicates**
   - 100% duplicates → LLM decides: halt_all_duplicates or proceed_anyway
   - >80% duplicates → LLM decides: proceed_with_warning or proceed_normal
   - Default fallback: halt_all_duplicates (all) / proceed_with_warning (mostly)

3. **After Story Writer — quality issues**
   - >50% need refinement → LLM decides: return_with_warning or return_normal
   - 0 epics formed → LLM decides: return_ungrouped or return_single_epic
   - Default fallback: return_with_warning / return_ungrouped

4. **On Permanent error**
   - LLM decides: return_partial (keep what succeeded) or halt_completely
   - Default fallback: return_partial if prior results exist

5. **Conflicts detected**
   - LLM decides: proceed_with_conflicts or add_conflict_summary
   - Default fallback: proceed_with_conflicts

### Key Implementation Details

- `ReActReasoner` class in `agents/react_reasoning.py` — stateless, takes LLMGenerationTool
- `reasoning_llm` parameter on OrchestratorAgent is **optional** — None means pure sequential pipeline (backward compatible)
- **`REACT_REASONING_ENABLED` env var** (default: `true`) — set to `false` to disable reasoning and save LLM calls. When false, `reasoning_llm=None` is passed regardless of available LLM tool.
- Every decision logged to AuditLog as "ReActReasoner" agent
- Every LLM call wrapped in try/except — failures never break the pipeline
- `SessionResult` has a `metadata` dict for ReAct notes (e.g., halt reasons)
- `main.py` passes the same LLM tool used by agents as the reasoning LLM (when enabled)

### Trigger Conditions (reasoning only fires on abnormal paths)

| Decision Point | Triggers When | Skips When |
|---|---|---|
| After Parser | 0 items extracted OR <3 items | ≥3 items and no errors |
| After Gap Detection | 100% duplicates OR >80% duplicates | Mixed classifications |
| After Story Writer | >50% need refinement OR 0 epics | ≤50% refinement AND epics > 0 |
| Permanent Error | Always | Never (always abnormal) |
| Conflicts Detected | >20% of items are conflicts | ≤20% conflict rate |

All skip/trigger decisions are logged to AuditLog for observability.
