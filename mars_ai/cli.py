"""Command line entry point for the Mars autonomous AI agent."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from mars_ai.agent import AutonomousMarsAgent, JsonMemoryStore, build_default_goals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mars-ai",
        description="Run a local self-directed AI agent focused on Mars research and mission planning.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Number of autonomous decision cycles to run.",
    )
    parser.add_argument(
        "--objective",
        help="Optional high-priority Mars objective to add to the agent's mission charter.",
    )
    parser.add_argument(
        "--memory",
        default=".mars-ai-memory.json",
        help="Path to JSON memory used across autonomous runs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text run report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = JsonMemoryStore(Path(args.memory))
    agent = AutonomousMarsAgent(goals=build_default_goals(args.objective), memory_store=store)
    entries = agent.run(cycles=args.cycles)

    if args.json:
        print(
            json.dumps(
                {
                    "entries": [asdict(entry) for entry in entries],
                    "status": agent.status(),
                },
                indent=2,
            )
        )
    else:
        _print_report(entries, agent.status(), Path(args.memory))

    return 0


def _print_report(entries: object, status: dict[str, object], memory_path: Path) -> None:
    print("Mars AI autonomous run")
    print(f"Memory: {memory_path}")
    print(f"Cycle: {status['cycle']}")
    print()

    for entry in entries:
        print(f"[cycle {entry.cycle}] {entry.focus_area}: {entry.task_description}")
        print(f"confidence: {entry.confidence:.2f}")
        print(entry.summary)
        if entry.follow_up_questions:
            print("follow-ups:")
            for question in entry.follow_up_questions:
                print(f"  - {question}")
        print()

    print(f"Backlog size: {len(status['backlog'])}")


if __name__ == "__main__":
    raise SystemExit(main())
