# Mars AI Agent

This repository contains a local-first scaffold for a self-directed AI agent
focused on Mars research, mission planning, and autonomous decision loops.

The agent can run without API keys or network access. It maintains a mission
charter, chooses its own next task, executes Mars-focused research/planning
steps from a curated knowledge base, records memory, and generates follow-up
work for future cycles.

## What it does

- Defines Mars mission goals with priorities and success metrics.
- Selects the next task autonomously from a generated backlog.
- Executes local Mars-focused research, planning, monitoring, experiment design,
  and reflection actions.
- Saves JSON memory so later runs continue from previous work.
- Explains each decision with a confidence score and follow-up questions.
- Provides an interactive prompt for questions, objectives, status, memory, and
  backlog inspection.
- Runs in a fully self-directed mode that assesses its own coverage,
  injects reflection tasks, closes evidence gaps, and persists every cycle.

## Quick start

Run the agent directly with Python:

```bash
python3 -m mars_ai --cycles 5
```

Add a high-priority Mars objective:

```bash
python3 -m mars_ai --objective "Find the safest lava tube habitat strategy" --cycles 3
```

Print JSON for another program to consume:

```bash
python3 -m mars_ai --cycles 2 --json
```

By default, memory is stored in `.mars-ai-memory.json`. Use a custom path when
you want separate autonomous runs:

```bash
python3 -m mars_ai --memory runs/habitat-memory.json --cycles 10
```

## Fully self-directed mode

Use `--self-directed` when you want the agent to keep choosing its own work
without prompts:

```bash
python3 -m mars_ai --self-directed --cycles 25
```

In this mode the agent:

1. Assesses focus-area coverage and average confidence.
2. Creates its own startup plan.
3. Adds periodic reflection tasks.
4. Adds evidence-gap tasks for high-priority goals.
5. Monitors low-confidence memory and schedules follow-up research.
6. Persists memory after each cycle.

For long-running use, provide a stop file path. The agent exits cleanly between
cycles when that file exists:

```bash
python3 -m mars_ai --self-directed --cycles 1000 --stop-file .mars-ai-stop
```

Then stop it from another terminal with:

```bash
touch .mars-ai-stop
```

You can also get the self-directed report as JSON:

```bash
python3 -m mars_ai --self-directed --cycles 10 --json
```

## Interactive mode

Start an interactive session when you want to talk to the agent directly:

```bash
python3 -m mars_ai --interactive
```

Inside the prompt, you can ask plain-language Mars questions or use commands:

```text
mars-ai> How can a Mars habitat reduce radiation risk?
mars-ai> objective Map water ice near a future landing site
mars-ai> auto 5
mars-ai> run 3
mars-ai> status
mars-ai> memory 5
mars-ai> backlog 10
mars-ai> quit
```

## Architecture

- `mars_ai.agent.Goal`: mission-level outcomes the agent pursues.
- `mars_ai.agent.Task`: concrete work selected by the autonomy loop.
- `mars_ai.agent.MemoryEntry`: durable record of completed work and uncertainty.
- `mars_ai.agent.SelfDirectedRunReport`: summary of an autonomous run, including
  stop reason and final autonomy state.
- `mars_ai.agent.JsonMemoryStore`: JSON persistence across runs.
- `mars_ai.agent.AutonomousMarsAgent`: self-directed loop that seeds goals,
  scores tasks, executes work, stores memory, and schedules follow-ups.
- `mars_ai.cli`: command line interface.

The current executor is deterministic and standard-library-only. A future
deployment can replace `MarsKnowledgeBase` or the task execution methods with
LLM calls, rover simulators, orbital data tools, or mission-control APIs while
keeping the same goal/task/memory contract.

## Test

```bash
python3 -m unittest discover -s tests
```
