"""Core autonomy loop for a Mars-focused AI agent.

The agent is intentionally local-first: it can run without API keys, network
access, or a model provider. A production deployment can replace the knowledge
base or task execution methods with LLM/tool integrations while keeping the
same goal, task, and memory loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Goal:
    """A mission-level outcome the agent should pursue autonomously."""

    name: str
    success_metric: str
    priority: int
    focus_area: str


@dataclass(frozen=True)
class Task:
    """A concrete piece of work selected by the agent."""

    id: str
    description: str
    focus_area: str
    action: str
    priority: int
    rationale: str


@dataclass(frozen=True)
class MemoryEntry:
    """A durable record of what the agent did and learned."""

    cycle: int
    task_id: str
    task_description: str
    focus_area: str
    summary: str
    confidence: float
    follow_up_questions: list[str]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class JsonMemoryStore:
    """Small JSON-backed memory store for autonomous runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[MemoryEntry]:
        if not self.path.exists():
            return []

        raw_entries = json.loads(self.path.read_text(encoding="utf-8"))
        return [MemoryEntry(**entry) for entry in raw_entries]

    def save(self, entries: Iterable[MemoryEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(entry) for entry in entries]
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class MarsKnowledgeBase:
    """Curated Mars facts used by the local task executor."""

    FACTS: dict[str, list[str]] = {
        "habitability": [
            "Mars has surface radiation, cold temperatures, and low pressure that make unshielded biology fragile.",
            "Subsurface ice, lava tubes, and regolith shielding are recurring habitat concepts.",
            "Closed-loop water, oxygen, and food systems are central to long-duration surface presence.",
        ],
        "resources": [
            "Water ice can support crew consumption, radiation shielding, propellant production, and agriculture.",
            "Atmospheric carbon dioxide can be processed for oxygen and methane production.",
            "In-situ resource utilization reduces launch mass but adds extraction, storage, and reliability risks.",
        ],
        "science": [
            "Astrobiology, geology, climate history, and subsurface mapping are high-value Mars science themes.",
            "Autonomous science agents should preserve provenance and confidence for every observation.",
            "Follow-up observations are most useful when they test a specific uncertainty.",
        ],
        "mobility": [
            "Dust, rocks, slopes, and communication delay make rover autonomy important on Mars.",
            "Safe traverse planning balances science value, energy, thermal constraints, and terrain risk.",
            "Orbital maps are useful but still require local hazard detection.",
        ],
        "mission": [
            "Mars communication delay prevents continuous real-time control from Earth.",
            "Robust autonomy needs explicit goals, safe operating bounds, memory, and explainable decisions.",
            "Mission plans should include fail-safe behavior when confidence is low.",
        ],
    }

    def research(self, focus_area: str, prompt: str, prior_memory_count: int) -> tuple[str, float, list[str]]:
        facts = self.FACTS.get(focus_area, self.FACTS["mission"])
        confidence = min(0.92, 0.62 + (0.04 * prior_memory_count))
        summary = (
            f"Investigated {focus_area} for: {prompt}. "
            f"Key Mars context: {' '.join(facts)} "
            f"Use this to refine the next autonomous decision."
        )
        follow_ups = [
            f"What evidence would most reduce uncertainty in {focus_area}?",
            f"Which operational constraint could block progress on {focus_area}?",
        ]
        return summary, confidence, follow_ups


class AutonomousMarsAgent:
    """Self-directed Mars agent with goals, task selection, execution, and memory."""

    def __init__(
        self,
        goals: Iterable[Goal] | None = None,
        memory_store: JsonMemoryStore | None = None,
        knowledge_base: MarsKnowledgeBase | None = None,
    ) -> None:
        self.goals = sorted(list(goals or build_default_goals()), key=lambda goal: goal.priority, reverse=True)
        self.memory_store = memory_store
        self.knowledge_base = knowledge_base or MarsKnowledgeBase()
        self.memory: list[MemoryEntry] = memory_store.load() if memory_store else []
        self.completed_task_ids = {entry.task_id for entry in self.memory}
        self.cycle = len(self.memory)
        self.backlog: list[Task] = []
        self._seed_backlog()

    def run(self, cycles: int = 5) -> list[MemoryEntry]:
        """Run a number of autonomous cycles and return new memory entries."""

        if cycles < 1:
            raise ValueError("cycles must be at least 1")

        new_entries = []
        for _ in range(cycles):
            new_entries.append(self.run_cycle())
        return new_entries

    def run_cycle(self) -> MemoryEntry:
        """Select, execute, remember, and expand the next task."""

        task = self._select_next_task()
        entry = self._execute(task)
        self.memory.append(entry)
        self.completed_task_ids.add(task.id)
        self.cycle += 1
        self._schedule_follow_ups(entry)

        if self.memory_store:
            self.memory_store.save(self.memory)

        return entry

    def status(self) -> dict[str, object]:
        """Return a serializable snapshot of the current autonomous state."""

        return {
            "cycle": self.cycle,
            "goals": [asdict(goal) for goal in self.goals],
            "memory_count": len(self.memory),
            "backlog": [asdict(task) for task in self.backlog],
            "completed_task_ids": sorted(self.completed_task_ids),
        }

    def _seed_backlog(self) -> None:
        for goal in self.goals:
            self._add_task(
                description=f"Research Mars context for goal: {goal.name}",
                focus_area=goal.focus_area,
                action="research",
                priority=goal.priority,
                rationale=f"Build baseline understanding for success metric: {goal.success_metric}",
            )
            self._add_task(
                description=f"Plan autonomous next steps for goal: {goal.name}",
                focus_area=goal.focus_area,
                action="plan",
                priority=goal.priority - 5,
                rationale="Convert mission goal into concrete self-directed work.",
            )

    def _select_next_task(self) -> Task:
        if not self.backlog:
            self._add_task(
                description="Reflect on mission memory and choose the next highest-value Mars investigation",
                focus_area="mission",
                action="reflect",
                priority=50,
                rationale="Keep the agent moving when no explicit task remains.",
            )

        self.backlog.sort(key=self._task_score, reverse=True)
        return self.backlog.pop(0)

    def _task_score(self, task: Task) -> tuple[int, int, int]:
        focus_count = sum(1 for entry in self.memory if entry.focus_area == task.focus_area)
        novelty_bonus = max(0, 8 - focus_count)
        action_bonus = {"research": 4, "monitor": 3, "plan": 2, "experiment": 1, "reflect": 0}.get(task.action, 0)
        return task.priority, novelty_bonus, action_bonus

    def _execute(self, task: Task) -> MemoryEntry:
        if task.action == "plan":
            summary, confidence, follow_ups = self._plan(task)
        elif task.action == "monitor":
            summary, confidence, follow_ups = self._monitor(task)
        elif task.action == "experiment":
            summary, confidence, follow_ups = self._experiment(task)
        elif task.action == "reflect":
            summary, confidence, follow_ups = self._reflect(task)
        else:
            summary, confidence, follow_ups = self.knowledge_base.research(
                task.focus_area,
                task.description,
                prior_memory_count=len(self.memory),
            )

        return MemoryEntry(
            cycle=self.cycle + 1,
            task_id=task.id,
            task_description=task.description,
            focus_area=task.focus_area,
            summary=summary,
            confidence=confidence,
            follow_up_questions=follow_ups,
        )

    def _plan(self, task: Task) -> tuple[str, float, list[str]]:
        related = [entry for entry in self.memory if entry.focus_area == task.focus_area]
        evidence_note = (
            f"{len(related)} prior memory entries are available for this focus area."
            if related
            else "No prior memory exists yet, so start with low-risk reconnaissance."
        )
        summary = (
            f"Planned autonomous work for {task.focus_area}. {evidence_note} "
            "Next sequence: gather evidence, identify operational constraints, run a small experiment, "
            "then reflect before expanding scope."
        )
        follow_ups = [
            f"Design a small experiment for {task.focus_area}.",
            f"Monitor mission risk tied to {task.focus_area}.",
        ]
        return summary, 0.74 if related else 0.66, follow_ups

    def _monitor(self, task: Task) -> tuple[str, float, list[str]]:
        summary = (
            f"Monitored Mars mission risk for {task.focus_area}. "
            "Primary checks: safety bounds, uncertainty, communications delay, energy use, and reversibility. "
            "Escalate to reflection if confidence drops below operational thresholds."
        )
        follow_ups = [
            f"What fail-safe should apply to {task.focus_area}?",
            "Reflect on whether current Mars priorities still match mission goals.",
        ]
        return summary, 0.78, follow_ups

    def _experiment(self, task: Task) -> tuple[str, float, list[str]]:
        summary = (
            f"Designed a bounded Mars autonomy experiment for {task.focus_area}. "
            "Hypothesis, required observations, stop conditions, and expected decision impact are recorded "
            "before any simulated execution."
        )
        follow_ups = [
            f"Research evidence needed to evaluate the {task.focus_area} experiment.",
            f"Monitor mission risk tied to {task.focus_area}.",
        ]
        return summary, 0.71, follow_ups

    def _reflect(self, task: Task) -> tuple[str, float, list[str]]:
        areas = sorted({entry.focus_area for entry in self.memory}) or ["mission"]
        summary = (
            f"Reflected on {len(self.memory)} memory entries across {', '.join(areas)}. "
            "The agent should prioritize high-priority goals with sparse evidence and keep each decision explainable."
        )
        follow_ups = [
            "Research the least-covered high-priority Mars goal.",
            "Plan the next autonomous Mars investigation based on current memory.",
        ]
        return summary, 0.69, follow_ups

    def _schedule_follow_ups(self, entry: MemoryEntry) -> None:
        for index, question in enumerate(entry.follow_up_questions):
            action = self._infer_action(question)
            self._add_task(
                description=question,
                focus_area=entry.focus_area if "mission" not in question.lower() else "mission",
                action=action,
                priority=max(20, 65 - (index * 6)),
                rationale=f"Generated after cycle {entry.cycle} to reduce uncertainty.",
            )

    def _add_task(
        self,
        description: str,
        focus_area: str,
        action: str,
        priority: int,
        rationale: str,
    ) -> None:
        task = Task(
            id=_task_id(action=action, focus_area=focus_area, description=description),
            description=description,
            focus_area=focus_area,
            action=action,
            priority=priority,
            rationale=rationale,
        )
        queued_ids = {queued.id for queued in self.backlog}
        if task.id not in queued_ids and task.id not in self.completed_task_ids:
            self.backlog.append(task)

    @staticmethod
    def _infer_action(text: str) -> str:
        lowered = text.lower()
        if "experiment" in lowered or "test" in lowered:
            return "experiment"
        if "monitor" in lowered or "risk" in lowered or "fail-safe" in lowered:
            return "monitor"
        if "plan" in lowered or "next" in lowered:
            return "plan"
        if "reflect" in lowered or "priorit" in lowered:
            return "reflect"
        return "research"


def build_default_goals(extra_objective: str | None = None) -> list[Goal]:
    """Create the default Mars mission charter."""

    goals = [
        Goal(
            name="Build a safe autonomous Mars mission loop",
            success_metric="Every decision records rationale, confidence, and follow-up uncertainty.",
            priority=95,
            focus_area="mission",
        ),
        Goal(
            name="Identify sustainable human-presence requirements",
            success_metric="Track habitat, resource, and safety constraints for long-duration presence.",
            priority=88,
            focus_area="habitability",
        ),
        Goal(
            name="Prioritize in-situ resource utilization",
            success_metric="Surface resource decisions connect to water, oxygen, propellant, or shielding needs.",
            priority=82,
            focus_area="resources",
        ),
        Goal(
            name="Advance Mars science autonomously",
            success_metric="Science tasks preserve provenance, uncertainty, and explicit next observations.",
            priority=76,
            focus_area="science",
        ),
        Goal(
            name="Improve rover and field mobility decisions",
            success_metric="Mobility plans balance terrain risk, energy, and science value.",
            priority=70,
            focus_area="mobility",
        ),
    ]

    if extra_objective:
        goals.append(
            Goal(
                name=extra_objective,
                success_metric="Convert the user objective into autonomous Mars tasks with recorded memory.",
                priority=100,
                focus_area="mission",
            )
        )

    return goals


def _task_id(action: str, focus_area: str, description: str) -> str:
    digest = hashlib.sha1(description.encode("utf-8")).hexdigest()[:10]
    return f"{action}-{focus_area}-{digest}"
