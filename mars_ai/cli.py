"""Command line entry point for the Mars autonomous AI agent."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Callable, TextIO

from mars_ai.agent import (
    AutonomousMarsAgent,
    JsonMemoryStore,
    MemoryEntry,
    SelfDirectedRunReport,
    build_default_goals,
)


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
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start an interactive session instead of running a fixed number of cycles.",
    )
    parser.add_argument(
        "--self-directed",
        action="store_true",
        help="Run the full self-directed autonomy loop.",
    )
    parser.add_argument(
        "--stop-file",
        help="Optional file path that cleanly stops a self-directed run when it exists.",
    )
    parser.add_argument(
        "--reflection-interval",
        type=int,
        default=4,
        help="How often self-directed mode injects a reflection task.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.interactive and args.self_directed:
        parser.error("--interactive and --self-directed cannot be used together")

    store = JsonMemoryStore(Path(args.memory))
    agent = AutonomousMarsAgent(goals=build_default_goals(args.objective), memory_store=store)

    if args.interactive:
        return run_interactive(agent, Path(args.memory))

    if args.self_directed:
        report = agent.run_self_directed(
            max_cycles=args.cycles,
            stop_file=args.stop_file,
            reflection_interval=args.reflection_interval,
        )
        if args.json:
            print(json.dumps(asdict(report), indent=2))
        else:
            _print_self_directed_report(report, Path(args.memory))
        return 0

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


def run_interactive(
    agent: AutonomousMarsAgent,
    memory_path: Path,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> int:
    """Run a small REPL for steering the Mars agent."""

    print("Mars AI interactive session", file=output)
    print(f"Memory: {memory_path}", file=output)
    print("Type 'help' for commands, or ask any Mars question.", file=output)
    print(file=output)

    while True:
        try:
            raw_command = input_fn("mars-ai> ")
        except EOFError:
            print(file=output)
            print("Session ended.", file=output)
            return 0
        except KeyboardInterrupt:
            print(file=output)
            print("Session interrupted.", file=output)
            return 130

        command = raw_command.strip()
        if not command:
            continue

        lowered = command.lower()
        if lowered in {"exit", "quit", "q"}:
            print("Session ended.", file=output)
            return 0

        if lowered in {"help", "?"}:
            _print_interactive_help(output)
            continue

        if lowered == "status":
            _print_status(agent.status(), output)
            continue

        if lowered == "goals":
            _print_goals(agent.status(), output)
            continue

        if lowered.startswith("backlog"):
            _print_backlog(agent.status(), output, _parse_optional_limit(command, default=8))
            continue

        if lowered.startswith("memory"):
            _print_memory(agent.memory, output, _parse_optional_limit(command, default=5))
            continue

        if lowered.startswith("auto"):
            count = _parse_optional_limit(command, default=5)
            report = agent.run_self_directed(max_cycles=count)
            _print_self_directed_report(report, memory_path, output)
            continue

        if lowered.startswith("run"):
            count = _parse_optional_limit(command, default=1)
            entries = agent.run(cycles=count)
            print(f"Ran {len(entries)} autonomous cycle(s).", file=output)
            for entry in entries:
                _print_entry(entry, output)
            continue

        if lowered.startswith("objective "):
            objective = command.partition(" ")[2].strip()
            if not objective:
                print("Please provide an objective after 'objective'.", file=output)
                continue
            goal = agent.add_goal(objective)
            print(f"Added objective: {goal.name} [{goal.focus_area}]", file=output)
            continue

        if lowered.startswith("ask "):
            prompt = command.partition(" ")[2].strip()
        else:
            prompt = command

        try:
            entry = agent.ask(prompt)
        except ValueError as error:
            print(str(error), file=output)
            continue
        _print_entry(entry, output)


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


def _print_self_directed_report(
    report: SelfDirectedRunReport,
    memory_path: Path,
    output: TextIO = sys.stdout,
) -> None:
    print("Mars AI self-directed run", file=output)
    print(f"Memory: {memory_path}", file=output)
    print(f"Stop reason: {report.stop_reason}", file=output)
    print(f"Cycles completed: {report.cycles_completed}", file=output)
    print(file=output)

    for entry in report.entries:
        _print_entry(entry, output)

    state = report.status.get("autonomy_state", {})
    if state:
        print("Autonomy state:", file=output)
        print(f"  average confidence: {state.get('average_confidence')}", file=output)
        print(f"  least-covered focus area: {state.get('least_covered_focus_area')}", file=output)
        print(f"  backlog size: {state.get('backlog_size')}", file=output)


def _print_interactive_help(output: TextIO) -> None:
    print("Commands:", file=output)
    print("  ask <question>       Ask the agent a Mars question", file=output)
    print("  <plain text>         Same as ask <plain text>", file=output)
    print("  run [n]              Let the agent run n autonomous cycles", file=output)
    print("  auto [n]             Run n self-directed autonomous cycles", file=output)
    print("  objective <text>     Add a new high-priority Mars objective", file=output)
    print("  status               Show cycle, memory, and backlog counts", file=output)
    print("  goals                Show mission goals", file=output)
    print("  backlog [n]          Show the next n queued tasks", file=output)
    print("  memory [n]           Show the latest n memory entries", file=output)
    print("  quit                 End the session", file=output)


def _print_entry(entry: MemoryEntry, output: TextIO) -> None:
    print(file=output)
    print(f"[cycle {entry.cycle}] {entry.focus_area}: {entry.task_description}", file=output)
    print(f"confidence: {entry.confidence:.2f}", file=output)
    print(entry.summary, file=output)
    if entry.follow_up_questions:
        print("follow-ups:", file=output)
        for question in entry.follow_up_questions:
            print(f"  - {question}", file=output)
    print(file=output)


def _print_status(status: dict[str, object], output: TextIO) -> None:
    print(f"Cycle: {status['cycle']}", file=output)
    print(f"Memory entries: {status['memory_count']}", file=output)
    print(f"Backlog tasks: {len(status['backlog'])}", file=output)
    print(f"Completed tasks: {len(status['completed_task_ids'])}", file=output)
    state = status.get("autonomy_state", {})
    if state:
        print(f"Average confidence: {state.get('average_confidence')}", file=output)
        print(f"Least-covered focus area: {state.get('least_covered_focus_area')}", file=output)


def _print_goals(status: dict[str, object], output: TextIO) -> None:
    for goal in status["goals"]:
        print(
            f"- [{goal['priority']}] {goal['name']} ({goal['focus_area']}): {goal['success_metric']}",
            file=output,
        )


def _print_backlog(status: dict[str, object], output: TextIO, limit: int) -> None:
    backlog = status["backlog"][:limit]
    if not backlog:
        print("Backlog is empty.", file=output)
        return

    for task in backlog:
        print(
            f"- [{task['priority']}] {task['action']} {task['focus_area']}: {task['description']}",
            file=output,
        )


def _print_memory(entries: list[MemoryEntry], output: TextIO, limit: int) -> None:
    if not entries:
        print("No memory entries yet.", file=output)
        return

    for entry in entries[-limit:]:
        print(
            f"- [cycle {entry.cycle}] {entry.focus_area}: {entry.task_description} "
            f"(confidence {entry.confidence:.2f})",
            file=output,
        )


def _parse_optional_limit(command: str, default: int) -> int:
    parts = command.split(maxsplit=1)
    if len(parts) == 1:
        return default

    try:
        parsed = int(parts[1])
    except ValueError:
        return default
    return max(1, parsed)


if __name__ == "__main__":
    raise SystemExit(main())
