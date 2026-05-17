from __future__ import annotations

from io import StringIO
import tempfile
import unittest
from pathlib import Path

from mars_ai.agent import AutonomousMarsAgent, JsonMemoryStore
from mars_ai.cli import run_interactive


class InteractiveCliTests(unittest.TestCase):
    def test_plain_text_prompt_runs_agent_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            agent = AutonomousMarsAgent(memory_store=JsonMemoryStore(memory_path))
            commands = iter(["How can Mars habitats reduce radiation risk?", "quit"])
            output = StringIO()

            exit_code = run_interactive(agent, memory_path, input_fn=lambda _: next(commands), output=output)

            self.assertEqual(exit_code, 0)
            self.assertIn("Mars AI interactive session", output.getvalue())
            self.assertIn("radiation", output.getvalue())
            self.assertEqual(agent.cycle, 1)

    def test_interactive_commands_show_status_and_add_objective(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            agent = AutonomousMarsAgent(memory_store=JsonMemoryStore(memory_path))
            commands = iter(["objective Study water ice extraction", "status", "goals", "quit"])
            output = StringIO()

            exit_code = run_interactive(agent, memory_path, input_fn=lambda _: next(commands), output=output)

            self.assertEqual(exit_code, 0)
            text = output.getvalue()
            self.assertIn("Added objective: Study water ice extraction [resources]", text)
            self.assertIn("Cycle: 0", text)
            self.assertIn("Study water ice extraction", text)


if __name__ == "__main__":
    unittest.main()
