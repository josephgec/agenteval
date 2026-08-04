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

import shlex
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
    "exec_edit_file",
    "exec_read_file",
    "exec_search",
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
    "Read a file from the workspace. Give start_line and line_count to read "
    "part of a large file.",
    path=P.str("Path to read."),
    start_line=P.num("First line to read, 1-based. Omit to start at the top.",
                     required=False),
    line_count=P.num("How many lines to read from start_line.", required=False),
)
def exec_read_file(
    world: World, path: str,
    start_line: float | None = None, line_count: float | None = None,
) -> str:
    """Returned verbatim, with no line numbers prefixed.

    Deliberate, and the reason `exec_edit_file` is usable: an agent edits by
    quoting back an exact snippet of what it read, and numbers woven into that
    text turn every edit into a de-numbering exercise. Navigation is what line
    numbers are for, and `exec_search` provides them there.
    """
    environment = _env(world)
    if start_line is None and line_count is None:
        result = environment.read_file(path)
    else:
        start = max(1, int(start_line or 1))
        end = start + int(line_count) - 1 if line_count else ""
        result = environment.exec(
            f"sed -n {shlex.quote(f'{start},{end or 0}p' if end else f'{start},$p')} "
            f"-- {shlex.quote(path)}"
        )
    if not result.ok:
        raise WorldError(f"could not read {path}: {result.stderr[:400]}")
    return result.stdout or "(empty file)"


@tool(
    "exec_edit_file",
    "Replace an exact snippet of a file with new text. `old_text` must appear "
    "exactly once — include surrounding lines to make it unique. Use this "
    "rather than rewriting a whole file.",
    path=P.str("File to edit."),
    old_text=P.str("The exact text to replace, copied from the file."),
    new_text=P.str("What to put in its place. Empty string deletes it."),
)
def exec_edit_file(world: World, path: str, old_text: str, new_text: str) -> str:
    """A targeted edit, because rewriting whole files is where agents lose.

    The two refusals below are the whole value of this tool over
    `exec_write_file`.

    *Not found is an error.* A silent no-op is the single most expensive
    failure mode available here: the agent believes it has made a change, every
    later step reasons from that belief, and the run ends with a confident
    summary of work that never happened. `str.replace` returning the string
    unchanged did exactly that to a change I made an hour before writing this.

    *Ambiguous is an error.* Replacing "the first occurrence" of a snippet that
    appears three times silently edits the wrong one, which is worse than not
    editing at all because it also corrupts something that was working.
    """
    environment = _env(world)
    result = environment.read_file(path)
    if not result.ok:
        raise WorldError(f"could not read {path}: {result.stderr[:400]}")
    content = result.stdout

    occurrences = content.count(old_text)
    if occurrences == 0:
        raise WorldError(
            f"that exact text is not in {path}. Whitespace and indentation "
            "have to match — read the file again and copy the snippet from it."
        )
    if occurrences > 1:
        raise WorldError(
            f"that text appears {occurrences} times in {path}, so replacing it "
            "would be ambiguous. Include the surrounding lines to pin down the "
            "one you mean."
        )

    updated = content.replace(old_text, new_text)
    written = environment.write_file(path, updated)
    world.record("exec", "edit_file", path,
                 removed=old_text.count("\n") + 1,
                 added=new_text.count("\n") + 1 if new_text else 0,
                 exit_code=written.exit_code)
    if not written.ok:
        raise WorldError(f"could not write {path}: {written.stderr[:400]}")
    line = content[:content.index(old_text)].count("\n") + 1
    return (
        f"Edited {path} at line {line}: replaced "
        f"{old_text.count(chr(10)) + 1} line(s) with "
        f"{new_text.count(chr(10)) + 1 if new_text else 0}."
    )


@tool(
    "exec_search",
    "Search file contents in the workspace and return matching lines with "
    "their file and line number.",
    pattern=P.str("Text or basic regular expression to search for."),
    path=P.str("Directory or file to search. Defaults to the workspace root.",
               required=False),
    max_results=P.num("Cap on matching lines returned (default 60).",
                      required=False),
)
def exec_search(
    world: World, pattern: str, path: str | None = None,
    max_results: float | None = None,
) -> str:
    """Finding the code to change, which is most of the work on a real repo.

    Line numbers *are* included here, unlike in `exec_read_file`: this output
    is for navigating to a place, not for copying text out of, so numbers help
    rather than getting in the way of a later edit.
    """
    limit = int(max_results or 60)
    target = path or "."
    # -I skips binaries, which in a checked-out repository is most of what a
    # naive grep would otherwise return from build directories.
    result = _env(world).exec(
        f"grep -rnI -- {shlex.quote(pattern)} {shlex.quote(target)} 2>&1 "
        f"| head -{limit}"
    )
    if not result.stdout.strip():
        # grep exits 1 on no matches, which is not a failure worth raising —
        # "nothing matched" is a useful answer and the agent should get it as
        # one rather than as an error to recover from.
        return f"No matches for {pattern!r} in {target}."
    return result.stdout


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
