from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mars_ai.agent import AutonomousMarsAgent, Goal, JsonMemoryStore, build_default_goals


class AutonomousMarsAgentTests(unittest.TestCase):
    def test_highest_priority_goal_drives_first_task(self) -> None:
        goals = [
            Goal(
                name="Low priority mobility check",
                success_metric="Document traverse risks.",
                priority=10,
                focus_area="mobility",
            ),
            Goal(
                name="Protect Mars habitat autonomy",
                success_metric="Record safety rationale.",
                priority=99,
                focus_area="habitability",
            ),
        ]
        agent = AutonomousMarsAgent(goals=goals)

        entry = agent.run_cycle()

        self.assertEqual(entry.focus_area, "habitability")
        self.assertIn("Protect Mars habitat autonomy", entry.task_description)
        self.assertGreater(entry.confidence, 0.6)

    def test_agent_generates_follow_up_work_after_each_cycle(self) -> None:
        agent = AutonomousMarsAgent(goals=build_default_goals())

        entry = agent.run_cycle()
        status = agent.status()

        self.assertEqual(entry.cycle, 1)
        self.assertGreaterEqual(len(entry.follow_up_questions), 1)
        self.assertGreater(len(status["backlog"]), 0)
        self.assertIn(entry.task_id, status["completed_task_ids"])

    def test_json_memory_store_persists_across_agent_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            store = JsonMemoryStore(memory_path)
            agent = AutonomousMarsAgent(memory_store=store)

            entries = agent.run(cycles=2)
            reloaded_agent = AutonomousMarsAgent(memory_store=JsonMemoryStore(memory_path))

            self.assertEqual(len(entries), 2)
            self.assertEqual(reloaded_agent.cycle, 2)
            self.assertEqual(len(reloaded_agent.memory), 2)

            persisted = json.loads(memory_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted[0]["task_id"], entries[0].task_id)

    def test_user_prompt_is_answered_as_high_priority_task(self) -> None:
        agent = AutonomousMarsAgent(goals=build_default_goals())

        entry = agent.ask("How should a rover choose a safe route across rocky terrain?")

        self.assertEqual(entry.focus_area, "mobility")
        self.assertIn("rover", entry.task_description)
        self.assertEqual(entry.cycle, 1)

    def test_runtime_goal_adds_new_goal_and_backlog_work(self) -> None:
        agent = AutonomousMarsAgent(goals=build_default_goals())

        goal = agent.add_goal("Map water ice near a future base")
        status = agent.status()

        self.assertEqual(goal.focus_area, "resources")
        self.assertEqual(status["goals"][0]["name"], "Map water ice near a future base")
        self.assertTrue(
            any("Map water ice near a future base" in task["description"] for task in status["backlog"])
        )

    def test_self_directed_run_creates_own_startup_plan_and_state(self) -> None:
        agent = AutonomousMarsAgent(goals=build_default_goals())

        report = agent.run_self_directed(max_cycles=3)

        self.assertEqual(report.cycles_completed, 3)
        self.assertEqual(report.stop_reason, "cycle_budget_reached")
        self.assertIn("Establish the autonomous Mars mission operating plan", report.entries[0].task_description)
        self.assertIn("autonomy_state", report.status)
        self.assertGreater(report.status["autonomy_state"]["backlog_size"], 0)

    def test_self_directed_run_stops_before_cycle_when_stop_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stop_file = Path(temp_dir) / "stop"
            stop_file.write_text("stop", encoding="utf-8")
            agent = AutonomousMarsAgent(goals=build_default_goals())

            report = agent.run_self_directed(max_cycles=5, stop_file=stop_file)

            self.assertEqual(report.cycles_completed, 0)
            self.assertEqual(report.stop_reason, "stop_file_detected")


if __name__ == "__main__":
    unittest.main()
