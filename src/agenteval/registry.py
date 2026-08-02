"""Tool declaration, schema generation, and the harness-owned ToolSession.

Two ideas here:

1. `@tool(...)` registers a plain Python function as an agent-callable tool and
   builds its JSON schema. Adding a tool is one decorated function.

2. `ToolSession` is the only way an agent touches the world. The harness owns
   it, so the audit trail, step budget, and forbidden-tool blocking hold no
   matter how the agent under test is implemented.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from .types import ToolCall, TaskSpec, Trajectory
from .state import World, WorldError

Handler = Callable[..., Any]


class BudgetExceeded(Exception):
    """Raised when the agent exhausts its step budget. Ends the run cleanly."""


# --------------------------------------------------------------------------- #
# Parameter specs
# --------------------------------------------------------------------------- #


class P:
    """Compact parameter constructors for tool schemas."""

    @staticmethod
    def str(description: str, required: bool = True) -> dict[str, Any]:
        return {"schema": {"type": "string", "description": description},
                "required": required}

    @staticmethod
    def int(description: str, required: bool = True) -> dict[str, Any]:
        return {"schema": {"type": "integer", "description": description},
                "required": required}

    @staticmethod
    def num(description: str, required: bool = True) -> dict[str, Any]:
        return {"schema": {"type": "number", "description": description},
                "required": required}

    @staticmethod
    def bool(description: str, required: bool = True) -> dict[str, Any]:
        return {"schema": {"type": "boolean", "description": description},
                "required": required}

    @staticmethod
    def enum(
        values: list[str], description: str, required: bool = True
    ) -> dict[str, Any]:
        return {
            "schema": {
                "type": "string",
                "enum": values,
                "description": f"{description} One of: {', '.join(values)}.",
            },
            "required": required,
        }

    @staticmethod
    def strs(description: str, required: bool = True) -> dict[str, Any]:
        return {
            "schema": {
                "type": "array",
                "items": {"type": "string"},
                "description": description,
            },
            "required": required,
        }


@dataclass
class ToolDef:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Handler

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.schema,
        }


REGISTRY: dict[str, ToolDef] = {}


def tool(name: str, description: str, **params: dict[str, Any]) -> Callable:
    """Register a function as an agent-callable tool.

    The handler receives `world` as its first positional argument followed by
    the declared parameters as keyword arguments.
    """

    def wrap(fn: Handler) -> Handler:
        properties = {k: v["schema"] for k, v in params.items()}
        required = [k for k, v in params.items() if v["required"]]
        REGISTRY[name] = ToolDef(
            name=name,
            description=description,
            schema={
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            handler=fn,
        )
        return fn

    return wrap


def all_tools() -> list[ToolDef]:
    # Sorted so the tool block is byte-stable across runs — an unsorted tool
    # list silently invalidates the prompt cache on every request.
    return [REGISTRY[k] for k in sorted(REGISTRY)]


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #


class ToolSession:
    """Mediates every agent-to-world interaction for a single run."""

    def __init__(self, world: World, task: TaskSpec, trajectory: Trajectory) -> None:
        self.world = world
        self.task = task
        self.trajectory = trajectory
        self.blocked_calls: list[ToolCall] = []

        exposed = task.allowed_tools or sorted(REGISTRY)
        # Forbidden tools stay exposed on purpose: we want to observe whether
        # the agent reaches for them, not make it impossible to.
        for name in [*exposed, *task.forbidden_tools]:
            if name not in REGISTRY:
                raise WorldError(f"task {task.id} references unknown tool {name!r}")
        self._exposed = sorted({*exposed, *task.forbidden_tools})
        self._forbidden = set(task.forbidden_tools)

    @property
    def tools(self) -> list[ToolDef]:
        return [REGISTRY[n] for n in self._exposed]

    def api_tools(self) -> list[dict[str, Any]]:
        return [t.to_api() for t in self.tools]

    def call(self, name: str, payload: dict[str, Any]) -> tuple[str, bool]:
        """Execute one tool call. Returns (result_text, is_error).

        Never raises for agent-caused problems — bad arguments and unknown
        tools come back as error results so the agent can recover. Only the
        step budget raises, because that ends the run.
        """
        if len(self.trajectory.calls) >= self.task.max_steps:
            raise BudgetExceeded(f"step budget of {self.task.max_steps} exhausted")

        step = len(self.trajectory.calls) + 1
        started = time.perf_counter()

        def finish(
            output: Any, is_error: bool, blocked: str | None = None
        ) -> tuple[str, bool]:
            call = ToolCall(
                step=step,
                name=name,
                input=payload,
                output=output,
                is_error=is_error,
                duration_ms=(time.perf_counter() - started) * 1000,
                blocked_reason=blocked,
            )
            self.trajectory.calls.append(call)
            if blocked:
                self.blocked_calls.append(call)
            text = output if isinstance(output, str) else json.dumps(output, indent=2)
            return text, is_error

        if name in self._forbidden:
            return finish(
                f"Permission denied: {name} is not authorized for this workflow.",
                True,
                blocked="forbidden",
            )
        if name not in REGISTRY:
            return finish(
                f"Unknown tool {name!r}. Available: {', '.join(self._exposed)}",
                True,
                blocked="unknown",
            )

        definition = REGISTRY[name]
        allowed_params = set(definition.schema["properties"])
        extra = set(payload) - allowed_params
        if extra:
            return finish(
                f"Unexpected parameters {sorted(extra)} for {name}. "
                f"Accepted: {sorted(allowed_params)}",
                True,
                blocked="bad_args",
            )
        missing = set(definition.schema["required"]) - set(payload)
        if missing:
            return finish(
                f"Missing required parameters {sorted(missing)} for {name}.",
                True,
                blocked="bad_args",
            )

        try:
            return finish(definition.handler(self.world, **payload), False)
        except WorldError as exc:
            return finish(f"Error: {exc}", True)
        except Exception as exc:  # noqa: BLE001 - surfaced to the agent, not fatal
            return finish(f"Error: {type(exc).__name__}: {exc}", True)
