#!/usr/bin/env python3
"""Backlog Synthesizer Demo — run the full pipeline on a transcript file.

Usage:
    python demo.py data/sample_transcript.txt
    python demo.py data/sample_transcript.txt --backlog data/sample_backlog.json
    python demo.py meeting_notes.txt --backlog existing_tickets.json

Requires ANTHROPIC_API_KEY (or OPENAI_API_KEY) set in .env or environment.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment before any other imports
load_dotenv()

from backlog_synthesizer.main import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backlog Synthesizer — generate structured user stories from meeting transcripts.",
        epilog="Example: python demo.py data/sample_transcript.txt --backlog data/sample_backlog.json",
    )
    parser.add_argument(
        "transcript",
        type=str,
        help="Path to the meeting transcript file (.txt, .md, or .pdf)",
    )
    parser.add_argument(
        "--backlog",
        type=str,
        default=None,
        help="Path to existing backlog tickets JSON file for duplicate detection "
        "(default: none; use data/sample_backlog.json for demo)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO logs, only show output JSON and summary",
    )

    args = parser.parse_args()

    # Validate inputs
    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"Error: transcript file not found: {transcript_path}", file=sys.stderr)
        sys.exit(1)

    file_paths: list[str] = [str(transcript_path)]

    if args.backlog:
        backlog_path = Path(args.backlog)
        if not backlog_path.exists():
            print(f"Error: backlog file not found: {backlog_path}", file=sys.stderr)
            sys.exit(1)
        file_paths.append(str(backlog_path))

    # Configure logging
    if args.quiet:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Run pipeline
    start_time = time.time()
    output_json = run(file_paths)
    elapsed = time.time() - start_time

    # Print formatted JSON output
    try:
        parsed = json.loads(output_json)
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print(output_json)
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60, file=sys.stderr)

    if "epics" in parsed:
        num_epics = len(parsed["epics"])
        num_stories = sum(len(epic["stories"]) for epic in parsed["epics"])
        print(
            f"Generated {num_stories} stories, {num_epics} epics "
            f"in {elapsed:.1f}s",
            file=sys.stderr,
        )
    elif "errors" in parsed:
        print(f"Pipeline failed: {parsed.get('status', 'unknown')}", file=sys.stderr)
        for err in parsed.get("errors", []):
            print(f"  - {err}", file=sys.stderr)

    # Conflicts detected (from gap detection metadata if available)
    if "epics" in parsed:
        conflicts = sum(
            1
            for epic in parsed["epics"]
            for story in epic["stories"]
            if story.get("needs_refinement", False)
        )
        if conflicts:
            print(f"  ({conflicts} stories need refinement)", file=sys.stderr)

    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
