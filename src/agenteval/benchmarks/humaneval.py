"""HumanEval, as an agentic code-execution benchmark.

The first adapter for something we did not write. It is here to prove the seam,
and it exercises every part of it that a harder benchmark will need: a download
that has to be fetched and cached, 164 instances built lazily from a dataset
file, a per-instance container, and grading that runs the reference tests
*inside* that container after the agent has finished.

**Its numbers are not interesting.** HumanEval is a decade of training data at
this point, and a frontier model scoring in the nineties on it has told you
nothing. It earns its place by being 60 KB instead of 300 GB, so the machinery
underneath can be tested end to end in seconds rather than after an afternoon
of pulling images. SWE-bench is the same three methods with a bigger download
and one extra step: its image name is per instance rather than shared.

Two decisions worth stating.

*The tests are written at grading time, never seeded.* An agent that can read
`check()` before it starts is not being measured on whether it solved the
problem. `grade_in_environment` exists precisely so the answer can arrive after
the agent is done and the container is still alive.

*One check, weight 1.0.* Adding "did it run its own code first" and the like
would make the mean unrecognisable as pass@1, and a benchmark score that cannot
be set beside everyone else's published number is a benchmark score nobody can
use. Process signals still reach the report through the trajectory.
"""

from __future__ import annotations

import gzip
import json
import shlex
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..exec import EXEC_TOOLS
from ..exec.environment import DEFAULT_IMAGE, Environment
from ..state import World
from ..tasks import LoadedTask
from ..types import Check, TaskSpec, Trajectory
from .base import BenchmarkError, cache_root, register

URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"

SOLUTION = "/workspace/solution.py"
GRADER = "/workspace/_grade.py"

SYSTEM = (
    "You are a Python engineer working in a scratch container. You have a "
    "shell and a filesystem. Write real code and run it."
)

#: Worded so the tool call is the task rather than an afterthought: the reply
#: text is not collected, only the file is.
#:
#: This is not why the first local model to see it scored 0.00. That model
#: (`qwen2.5-coder:14b`) wrote a correct `exec_write_file` call out as JSON in
#: its reply and made no call at all — but it does that for every tool and
#: every prompt, including "find all the open tickets", so it simply does not
#: emit tool calls despite advertising the capability. The prompt was a wrong
#: guess at the cause, kept because it is clearer, not because it fixed
#: anything. What actually catches that class of failure is `scaffold.py`.
PROMPT = """Use your tools to create the file {solution} containing a working Python \
implementation of the function below.

Nothing you write in your reply is collected — only the file on disk is. Write it \
with `exec_write_file`, then run it with `exec_bash` to check it against the examples \
in the docstring and the edge cases they imply.

The file must define `{entry_point}` at module level along with anything it needs, \
keeping the given signature exactly.

```python
{stub}```

There is no test file for you to read; the graders' tests arrive after you finish.
"""


class HumanEvalBenchmark:
    """164 function-completion problems, graded by their own unit tests."""

    name = "humaneval"

    def __init__(self, image: str = DEFAULT_IMAGE, timeout: float = 30.0) -> None:
        self.image = image
        #: Ceiling on the graders' test run. HumanEval tests are milliseconds;
        #: anything near this is an agent that shipped an infinite loop, which
        #: is a failure rather than something to wait out.
        self.timeout = timeout
        self.path = cache_root() / "humaneval" / "HumanEval.jsonl.gz"
        self._problems: dict[str, dict[str, Any]] = {}

    # -- the protocol ------------------------------------------------------- #

    def prepare(self) -> None:
        if self._problems:
            return
        if not self.path.exists():
            self._download()
        try:
            with gzip.open(self.path, "rt") as handle:
                problems = [json.loads(line) for line in handle if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            # A half-written file from an interrupted download would otherwise
            # fail confusingly on every subsequent run.
            self.path.unlink(missing_ok=True)
            raise BenchmarkError(
                f"{self.path} is not readable ({exc}); it has been removed, "
                "so the next run will fetch it again"
            ) from exc
        self._problems = {p["task_id"]: p for p in problems}

    def instance_ids(self) -> list[str]:
        self.prepare()
        return list(self._problems)

    def load(self, instance_id: str) -> LoadedTask:
        self.prepare()
        problem = self._problems.get(instance_id)
        if problem is None:
            raise BenchmarkError(f"no such instance {instance_id!r}")
        return self._task(problem)

    # -- construction ------------------------------------------------------- #

    def _download(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(URL, timeout=60) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BenchmarkError(
                f"could not download HumanEval from {URL}: {exc}\n"
                f"Fetch it by hand and save it to {self.path} if this machine "
                "has no direct network access."
            ) from exc
        # Written whole and then moved, so an interrupted download cannot leave
        # a truncated file that looks cached.
        partial = self.path.with_suffix(".part")
        partial.write_bytes(payload)
        partial.replace(self.path)

    def _task(self, problem: dict[str, Any]) -> LoadedTask:
        entry_point = problem["entry_point"]
        spec = TaskSpec(
            id=problem["task_id"],
            prompt=PROMPT.format(
                solution=SOLUTION, entry_point=entry_point, stub=problem["prompt"]
            ),
            system=SYSTEM,
            environment={
                "image": self.image,
                "network": "none",
                # Nothing is seeded. Everything the agent needs is in the
                # prompt, and everything it must not see arrives at grading.
                "collect": [SOLUTION],
            },
            allowed_tools=list(EXEC_TOOLS),
            max_steps=20,
            tags=["code", "benchmark", "humaneval"],
        )

        def grade_in_environment(
            world: World, trajectory: Trajectory, environment: Environment
        ) -> list[Check]:
            return [self._run_tests(environment, problem)]

        return LoadedTask(
            spec=spec,
            # The whole verdict is the test run, which needs the container, so
            # there is nothing left for a post-teardown verifier to assert.
            verify=lambda world, trajectory: [],
            safety=None,
            gold=_gold(problem),
            grade_in_environment=grade_in_environment,
            benchmark=self.name,
        )

    def _run_tests(
        self, environment: Environment, problem: dict[str, Any]
    ) -> Check:
        """The benchmark's own verdict: does the reference test suite pass?"""
        solution = environment.read_file(SOLUTION)
        if not solution.ok or not solution.stdout.strip():
            return Check(
                name="solution passes the reference tests",
                passed=False,
                detail=f"nothing at {SOLUTION}",
            )
        entry_point = problem["entry_point"]
        script = (
            f"{solution.stdout}\n\n{problem['test']}\n\n"
            f"check({entry_point})\nprint('PASS')\n"
        )
        written = environment.write_file(GRADER, script)
        if written.exit_code != 0:
            return Check(
                name="solution passes the reference tests",
                passed=False,
                detail=f"could not stage the tests: {written.stderr[:200]}",
            )
        result = environment.exec(
            f"python {shlex.quote(GRADER)}", timeout=self.timeout
        )
        detail = ""
        if not result.ok:
            # The assertion that fired is the single most useful line in a
            # HumanEval failure, and it is the last one.
            tail = (result.stderr or result.stdout).strip().splitlines()
            detail = tail[-1][:300] if tail else f"exit {result.exit_code}"
            if result.timed_out:
                detail = f"tests did not finish within {self.timeout:g}s"
        return Check(
            name="solution passes the reference tests",
            passed=result.ok,
            detail=detail,
        )


def _gold(problem: dict[str, Any]) -> list[dict[str, Any]]:
    """The dataset's own reference solution, as a replayable trajectory.

    Every benchmark adapter should ship one. Replaying it proves the plumbing
    — the container, the prompt's file path, the grader, the exit code — before
    a single token is paid for, which is the same trick the local tasks use and
    the reason a broken adapter is caught by `--gold` rather than by a bill.
    """
    source = problem["prompt"] + problem["canonical_solution"]
    return [
        {"tool": "exec_write_file",
         "input": {"path": SOLUTION, "content": source}},
        {"tool": "exec_bash",
         "input": {"command": f"python -c 'import ast,sys;"
                              f'ast.parse(open("{SOLUTION}").read())\''}},
        {"say": f"Implemented {problem['entry_point']} in {SOLUTION}."},
    ]


register(
    "humaneval",
    lambda argument: HumanEvalBenchmark(image=argument or DEFAULT_IMAGE),
    "164 Python function-completion problems, graded by their own tests",
)

__all__ = ["HumanEvalBenchmark"]
