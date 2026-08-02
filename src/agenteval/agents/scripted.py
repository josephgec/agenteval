"""A deterministic agent that replays a fixed list of tool calls.

Two uses:

* **Gold trajectories.** Each task ships the call sequence a correct solution
  makes. The test suite runs it and asserts the verifier scores 1.0, which
  catches verifiers that are broken or trivially unsatisfiable — a failure mode
  that otherwise hides until you have spent real money discovering that every
  model "fails" an impossible task.
* **Offline development.** The whole pipeline — world, session, verifier,
  scoring, reporting — runs with no API key and no cost.
"""

from __future__ import annotations

from typing import Any

from ..registry import BudgetExceeded, ToolSession
from ..types import TaskSpec, Trajectory

#: A step is either {"tool": name, "input": {...}} or {"say": "final text"}.
Step = dict[str, Any]


class ScriptedAgent:
    model = None

    def __init__(self, script: list[Step], name: str = "scripted") -> None:
        self.script = script
        self.name = name

    async def run(
        self, task: TaskSpec, session: ToolSession, trajectory: Trajectory
    ) -> None:
        trajectory.turns = 1
        for step in self.script:
            if "say" in step:
                trajectory.messages.append(step["say"])
                continue
            try:
                session.call(step["tool"], step.get("input", {}))
            except BudgetExceeded as exc:
                trajectory.error = str(exc)
                break
        trajectory.stop_reason = "end_turn"
        trajectory.final_text = (
            trajectory.messages[-1] if trajectory.messages else ""
        )
