"""The agent-under-test interface.

Anything that can drive a ToolSession to completion is evaluable. Implement
`run`, register the class, and it competes on the same tasks as the reference
Claude agent with the same grading and the same audit trail.

The contract is narrow on purpose:

* You get the task and a session. You do not get the World — every state change
  must go through a recorded tool call.
* You fill in `trajectory.final_text` (and optionally `messages`/`thinking`).
  The harness fills in usage, timing, and the call log.
* Raising is allowed. The runner catches it and records `agent_error`; you do
  not need your own try/except for reporting purposes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..registry import ToolSession
from ..types import TaskSpec, Trajectory


@runtime_checkable
class Agent(Protocol):
    #: Stable identifier used in reports and result filenames.
    name: str
    #: Model id, when the agent is backed by one. Drives cost accounting.
    model: str | None

    async def run(
        self, task: TaskSpec, session: ToolSession, trajectory: Trajectory
    ) -> None:
        """Work the task to completion, calling `session.call(name, input)`."""
        ...
