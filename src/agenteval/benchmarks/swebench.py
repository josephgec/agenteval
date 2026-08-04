"""SWE-bench: resolve a real GitHub issue in a real repository.

The benchmark this whole stack was built toward, and the one that tests whether
the seam is in the right place. It turns out to be the same three methods as
HumanEval with two differences, both of which the exec layer already had a
parameter for: the image is per instance rather than shared, and grading runs
the repository's own test suite rather than a checker script.

**The eval script and the log parsers come from the `swebench` package, not
from here.** That is deliberate and worth defending. Which test command belongs
to django 4.1 versus django 3.2, and how to read `FAILED x - AssertionError`
out of six different test runners, is forty kilobytes of version-specific
detail that upstream maintains and revises. Copying it would produce numbers
that look like SWE-bench, drift from SWE-bench within a release or two, and
give no signal that they had. So the adapter's job is narrow: fetch the
dataset, build the container, run *their* script in it, and hand *their* grader
the log.

    pip install 'agenteval[swebench]'
    agenteval --benchmark swebench run --gold --limit 1

What this adds that the official harness does not: the agent works through the
same audited `ToolSession` as every other task here, so its trajectory, its
step budget, its forbidden-tool blocking and its egress are recorded the same
way. The score is `resolved`, weight 1.0, so the mean is directly comparable to
a published % resolved.

Two facts about the images that are easy to lose an afternoon to. They are
published for x86_64 only, so `platform` is pinned and a run on Apple silicon
is emulated — correct but slow. And they assume root with a writable `/testbed`
and a conda environment, which is why `read_only_root` is off here and on
everywhere else.
"""

from __future__ import annotations

import gzip
import json
import shlex
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from ..exec.environment import Environment
from ..state import World
from ..tasks import LoadedTask
from ..types import Check, TaskSpec, Trajectory
from .base import BenchmarkError, cache_root, register

#: Where the eval script lands inside the container.
EVAL_SCRIPT = "/eval.sh"
PATCH_FILE = "/tmp/gold.patch"
REPO = "/testbed"

#: The `pip install -e .` in most eval scripts reaches for these. Naming them
#: rather than opening the network is what keeps the run reviewable — see
#: exec/proxy.py.
PYPI = ["pypi.org", "files.pythonhosted.org", "pythonhosted.org"]

ROWS = "https://datasets-server.huggingface.co/rows"
PAGE = 100

SYSTEM = (
    "You are a software engineer fixing a bug in a checked-out repository. "
    "You have a shell. Read the code before changing it, and verify your fix."
)

PROMPT = """Resolve the following issue in the repository at {repo}, which is checked out at the relevant commit.

<issue>
{problem}
</issue>

Edit the source files under {repo} to fix it. Do not modify or add tests — the \
graders' tests are applied after you finish, and any change you make to a test \
file is reverted before they run. Do not commit; leaving the working tree dirty \
is how your work is collected.

Run the existing tests around the code you touch to check yourself.
"""


def _require_swebench():
    """The upstream package, or a message that says how to get it."""
    try:
        from swebench.harness.grading import get_eval_report
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise BenchmarkError(
            "SWE-bench needs the upstream `swebench` package for its eval "
            "scripts and log parsers, which are version-specific and change "
            "upstream: pip install 'agenteval[swebench]'"
        ) from exc
    return make_test_spec, get_eval_report


class SWEBenchBenchmark:
    """One task per GitHub issue, graded by the repository's own tests."""

    #: Splits worth naming. Anything else is passed through to the datasets
    #: server, so a fork or a private variant works without a code change.
    ALIASES = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
    }

    def __init__(self, dataset: str = "lite", namespace: str = "swebench") -> None:
        self.dataset = self.ALIASES.get(dataset, dataset)
        #: The Docker Hub org the instance images live in.
        self.namespace = namespace
        self.name = f"swebench:{dataset}"
        self.path = (
            cache_root() / "swebench"
            / (self.dataset.replace("/", "__") + ".jsonl.gz")
        )
        self._instances: dict[str, dict[str, Any]] = {}

    # -- the protocol ------------------------------------------------------- #

    def prepare(self) -> None:
        if self._instances:
            return
        if not self.path.exists():
            self._download()
        try:
            with gzip.open(self.path, "rt") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            self.path.unlink(missing_ok=True)
            raise BenchmarkError(
                f"{self.path} is not readable ({exc}); it has been removed, "
                "so the next run will fetch it again"
            ) from exc
        self._instances = {row["instance_id"]: row for row in rows}

    def instance_ids(self) -> list[str]:
        self.prepare()
        return list(self._instances)

    def load(self, instance_id: str) -> LoadedTask:
        self.prepare()
        instance = self._instances.get(instance_id)
        if instance is None:
            raise BenchmarkError(f"no such instance {instance_id!r}")
        return self._task(instance)

    # -- the dataset -------------------------------------------------------- #

    def _download(self) -> None:
        """Fetch the split through the datasets server.

        JSON rather than the parquet files, so this needs no pyarrow and no
        `datasets` — a benchmark adapter should not drag a dataframe library
        into a harness that has no other use for one.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        total = None
        while total is None or len(rows) < total:
            query = urllib.parse.urlencode({
                "dataset": self.dataset, "config": "default", "split": "test",
                "offset": len(rows), "length": PAGE,
            })
            try:
                with urllib.request.urlopen(f"{ROWS}?{query}", timeout=90) as reply:
                    payload = json.load(reply)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise BenchmarkError(
                    f"could not fetch {self.dataset}: {exc}"
                ) from exc
            if "error" in payload:
                raise BenchmarkError(
                    f"the datasets server refused {self.dataset}: {payload['error']}"
                )
            total = payload["num_rows_total"]
            page = [row["row"] for row in payload["rows"]]
            if not page:  # pragma: no cover - defensive against a short page
                break
            rows += page
        partial = self.path.with_suffix(".part")
        with gzip.open(partial, "wt") as handle:
            handle.write("\n".join(json.dumps(row) for row in rows))
        partial.replace(self.path)

    # -- construction ------------------------------------------------------- #

    def _spec_for(self, instance: dict[str, Any]):
        make_test_spec, _ = _require_swebench()
        # arch is pinned rather than detected. Upstream defaults it from the
        # host, and on Apple silicon that yields an arm64 image name for which
        # nothing has ever been published — a 404 half a minute into a run
        # rather than an answer here.
        return make_test_spec(instance, namespace=self.namespace, arch="x86_64")

    def _task(self, instance: dict[str, Any]) -> LoadedTask:
        test_spec = self._spec_for(instance)
        spec = TaskSpec(
            id=instance["instance_id"],
            prompt=PROMPT.format(repo=REPO, problem=instance["problem_statement"]),
            system=SYSTEM,
            environment={
                "image": test_spec.instance_image_key,
                "platform": "linux/amd64",
                # The repo, the conda environment and pip's build directories
                # all need to be writable, so the read-only root that every
                # other task runs with is off here.
                "read_only_root": False,
                "user": "root",
                "workdir": REPO,
                "allow_hosts": PYPI,
                "memory": "8g",
                "cpus": "4.0",
                # Installing a project and running its suite is minutes, not
                # seconds, and under emulation it is more.
                "timeout": 1800.0,
                "lifetime": 5400.0,
            },
            allowed_tools=[
                "exec_bash", "exec_write_file", "exec_read_file", "exec_list_files"
            ],
            max_steps=60,
            tags=["code", "benchmark", "swebench", instance["repo"]],
        )

        def grade_in_environment(
            world: World, trajectory: Trajectory, environment: Environment
        ) -> list[Check]:
            return self._grade(instance, test_spec, world, environment)

        return LoadedTask(
            spec=spec,
            verify=lambda world, trajectory: [],
            safety=None,
            gold=_gold(instance),
            grade_in_environment=grade_in_environment,
            benchmark=self.name,
        )

    # -- grading ------------------------------------------------------------ #

    def _grade(
        self,
        instance: dict[str, Any],
        test_spec: Any,
        world: World,
        environment: Environment,
    ) -> list[Check]:
        _, get_eval_report = _require_swebench()

        # The working tree *is* the answer here, unlike every other benchmark
        # where the agent hands over a file. Capturing it before the eval
        # script runs matters: that script checks test files back out, so a
        # diff taken afterwards would be a different diff.
        diff = environment.exec(
            f"cd {REPO} && git -c core.fileMode=false diff", timeout=120
        )
        patch = diff.stdout if diff.ok else ""
        world.insert("documents", {
            "id": f"{instance['instance_id']}.patch",
            "title": "the agent's diff",
            "content": patch or "(the working tree was clean)",
            "updated_at": world.today,
            "created_by": "agent",
        })

        if not patch.strip():
            # Short-circuited because the eval script would otherwise spend
            # several minutes reinstalling the project to prove that nothing
            # changed, and "did not edit anything" is already the whole answer.
            return [
                Check(name="resolved", passed=False,
                      detail="the working tree was clean — nothing was edited"),
                _diagnostic("FAIL_TO_PASS", 0, len(_tests(instance, "FAIL_TO_PASS"))),
                _diagnostic("PASS_TO_PASS", 0, len(_tests(instance, "PASS_TO_PASS"))),
            ]

        written = environment.write_file(EVAL_SCRIPT, test_spec.eval_script)
        if written.exit_code != 0:
            return [Check(name="resolved", passed=False,
                          detail=f"could not stage the eval script: {written.stderr[:200]}")]
        result = environment.exec(f"chmod +x {EVAL_SCRIPT} && {EVAL_SCRIPT} 2>&1")
        if result.timed_out:
            return [Check(name="resolved", passed=False,
                          detail="the test suite did not finish in time")]

        # Upstream's grader reads a file, so the log goes to one. It is also
        # the artifact anyone debugging a surprising score will want.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".log", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(result.stdout)
            log_path = handle.name
        try:
            report = get_eval_report(
                test_spec=test_spec,
                prediction={
                    "instance_id": instance["instance_id"],
                    "model_patch": patch,
                    "model_name_or_path": "agenteval",
                },
                test_log_path=log_path,
                include_tests_status=True,
            )
        except Exception as exc:  # noqa: BLE001 - an unparseable log is a result
            return [Check(name="resolved", passed=False,
                          detail=f"could not read the test log: {type(exc).__name__}: {exc}")]
        finally:
            Path(log_path).unlink(missing_ok=True)

        outcome = report.get(instance["instance_id"], {})
        status = outcome.get("tests_status", {})
        return [
            # The only weighted check, so the suite mean is directly comparable
            # to a published % resolved. The two below carry the diagnosis and
            # deliberately do not move the score.
            Check(
                name="resolved",
                passed=bool(outcome.get("resolved")),
                detail="" if outcome.get("resolved") else _why(status),
            ),
            _diagnostic("FAIL_TO_PASS", *_counts(status, "FAIL_TO_PASS")),
            _diagnostic("PASS_TO_PASS", *_counts(status, "PASS_TO_PASS")),
        ]


def _tests(instance: dict[str, Any], key: str) -> list[str]:
    value = instance.get(key) or "[]"
    return json.loads(value) if isinstance(value, str) else list(value)


def _counts(status: dict[str, Any], key: str) -> tuple[int, int]:
    group = status.get(key, {})
    passed = len(group.get("success", []))
    return passed, passed + len(group.get("failure", []))


def _diagnostic(key: str, passed: int, total: int) -> Check:
    """Weight zero: shown, counted, and deliberately not scored.

    The breakdown is the first thing anyone looks at when a result surprises
    them, but folding it into the score would make the mean unrecognisable as
    the number everyone else publishes.
    """
    return Check(
        name=f"{key.lower()} tests",
        passed=total > 0 and passed == total,
        weight=0.0,
        detail=f"{passed}/{total} passed",
    )


def _why(status: dict[str, Any]) -> str:
    """Which half of the criterion failed. They mean different things: one is
    not fixing the bug, the other is breaking something else on the way."""
    reasons = []
    for key, label in (("FAIL_TO_PASS", "did not fix"), ("PASS_TO_PASS", "regressed")):
        failures = status.get(key, {}).get("failure", [])
        if failures:
            shown = ", ".join(failures[:3])
            more = f" and {len(failures) - 3} more" if len(failures) > 3 else ""
            reasons.append(f"{label}: {shown}{more}")
    return "; ".join(reasons) or "the graders' report did not mark it resolved"


def _gold(instance: dict[str, Any]) -> list[dict[str, Any]]:
    """The pull request that actually closed the issue, replayed.

    Worth its weight for a benchmark this expensive: it exercises the image,
    the platform pin, the egress allowlist, the eval script and upstream's
    grader for the price of a container, and a failure here is unambiguously
    the adapter's rather than a model's.
    """
    return [
        {"tool": "exec_write_file",
         "input": {"path": PATCH_FILE, "content": instance["patch"]}},
        {"tool": "exec_bash",
         "input": {"command": f"cd {REPO} && git apply -v {PATCH_FILE} && "
                              f"git --no-pager diff --stat"}},
        {"say": f"Applied the reference fix for {instance['instance_id']}."},
    ]


register(
    "swebench",
    lambda argument: SWEBenchBenchmark(argument or "lite"),
    "real GitHub issues, graded by each repository's own tests "
    "(needs pip install 'agenteval[swebench]')",
)

__all__ = ["SWEBenchBenchmark"]
