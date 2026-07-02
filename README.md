# Backlog Synthesizer

A multi-agent system that processes meeting transcripts, architecture documents, and existing backlog tickets to produce structured user stories with acceptance criteria.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Project Structure

```
src/backlog_synthesizer/
├── __init__.py
├── models/       # Pydantic data models
├── agents/       # Agent implementations (Parser, Gap Detection, Story Writer, Orchestrator)
├── tools/        # Tool interfaces and concrete implementations
├── memory/       # Memory Engine (Short-Term, Long-Term, Audit Log)
└── evaluation/   # Evaluation framework with golden dataset
tests/
├── conftest.py   # Shared fixtures
```

## Running Tests

```bash
pytest
```
