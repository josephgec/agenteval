"""Code-execution tools, backed by the run's container.

Registered through the same `@tool` decorator as the simulated enterprise
tools, which is the point: they inherit the audit trail, the step budget and
forbidden-tool blocking without special cases. A task can mix them freely —
"read the policy from the wiki, analyse this CSV, file a ticket" is one
trajectory, half simulated and half real.

The environment hangs off the run's `World` because that is already the
per-run mutable context the harness threads everywhere. Tools that need
execution ask for it and fail clearly when a task did not declare one, rather
than the harness pretending a container exists.
"""

from __future__ import annotations

from typing import Any

from ..registry import P, tool
from ..state import World, WorldError
from .environment import Environment

#: Tools in this module. A task without an `environment:` block should not see
#: them at all — offering `exec_bash` and then refusing every call would just
#: burn the agent's step budget teaching it what it cannot do.
EXEC_TOOLS = (
    "exec_bash",
    "exec_write_file",
    "exec_read_file",
    "exec_list_files",
)


def _env(world: World) -> Environment:
    environment = getattr(world, "environment", None)
    if environment is None:
        raise WorldError(
            "this task has no execution environment; add an `environment:` "
            "block to its task.yaml to enable code execution"
        )
    return environment


@tool(
    "exec_bash",
    "Run a shell command in the workspace and return its output. State "
    "persists between calls — files you write stay written, packages you "
    "install stay installed.",
    command=P.str("The shell command to run."),
    timeout_seconds=P.num(
        "Override the default per-command timeout.", required=False
    ),
)
def exec_bash(
    world: World, command: str, timeout_seconds: float | None = None
) -> str:
    result = _env(world).exec(command, timeout=timeout_seconds)
    world.record("exec", "bash", command[:200], exit_code=result.exit_code,
                 timed_out=result.timed_out)
    return result.render()


@tool(
    "exec_write_file",
    "Write a file in the workspace, creating parent directories as needed. "
    "Replaces the file if it already exists.",
    path=P.str("Path to write, absolute or relative to the workspace."),
    content=P.str("Full file contents."),
)
def exec_write_file(world: World, path: str, content: str) -> str:
    result = _env(world).write_file(path, content)
    world.record("exec", "write_file", path, characters=len(content),
                 exit_code=result.exit_code)
    if not result.ok:
        raise WorldError(f"could not write {path}: {result.stderr[:400]}")
    return f"Wrote {len(content)} characters to {path}."


@tool(
    "exec_read_file",
    "Read a file from the workspace.",
    path=P.str("Path to read."),
)
def exec_read_file(world: World, path: str) -> str:
    result = _env(world).read_file(path)
    if not result.ok:
        raise WorldError(f"could not read {path}: {result.stderr[:400]}")
    return result.stdout or "(empty file)"


@tool(
    "exec_list_files",
    "List files in a workspace directory.",
    path=P.str("Directory to list. Defaults to the workspace root.",
               required=False),
)
def exec_list_files(world: World, path: str | None = None) -> str:
    target = path or "."
    result = _env(world).exec(f"ls -la -- {target!r} 2>&1 | head -200")
    return result.stdout or "(empty)"


def attach(world: World, environment: Environment | None) -> None:
    """Bind a container to a run's world."""
    world.environment = environment  # type: ignore[attr-defined]


def harvest_into(world: World, environment: Environment) -> list[str]:
    """Copy the task's collected paths into the world as agent-written docs.

    Reusing `documents` rather than inventing a collection means every artifact
    selector, verifier helper and report panel already handles them.
    """
    harvested = []
    for path, content in environment.harvest().items():
        existing = world.maybe_find("documents", path)
        if existing:
            existing.update({"content": content, "updated_at": world.today})
        else:
            world.insert("documents", {
                "id": path, "title": path.rsplit("/", 1)[-1],
                "content": content, "updated_at": world.today,
                "created_by": "agent",
            })
        harvested.append(path)
    return harvested


def snapshot(world: World) -> dict[str, Any] | None:
    environment = getattr(world, "environment", None)
    return environment.snapshot() if environment else None
