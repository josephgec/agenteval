"""Benchmarks: where tasks come from.

Three layers, tested separately. The protocol and selection are pure and need
nothing. The adapters are tested against a fixture rather than the network, so
the suite does not depend on GitHub being up. The end-to-end tests need the
exec image and skip cleanly without it.
"""

import asyncio
import gzip
import json

import pytest

from agenteval import RunConfig, ScriptedAgent, TaskSpec, Trajectory, World, run_one
from agenteval.benchmarks import (
    Benchmark,
    BenchmarkError,
    HumanEvalBenchmark,
    LocalBenchmark,
    base,
    cache_root,
    load_tasks,
    register,
    registered,
    resolve,
    select,
    summarise,
)
from agenteval.exec import environment as env_mod
from agenteval.tasks import DEFAULT_TASK_ROOT, LoadedTask
from agenteval.types import Check

IMAGE = env_mod.DEFAULT_IMAGE


# --------------------------------------------------------------------------- #
# The protocol
# --------------------------------------------------------------------------- #


def test_both_adapters_satisfy_the_protocol():
    """Structural, not inherited. A benchmark adapter should be able to live in
    someone else's package without importing a base class from ours."""
    assert isinstance(LocalBenchmark(), Benchmark)
    assert isinstance(HumanEvalBenchmark(), Benchmark)


def test_a_benchmark_is_three_methods():
    """The surface a new adapter has to implement, asserted so it cannot grow
    quietly — every method added here is work imposed on every future one."""

    class Minimal:
        name = "minimal"

        def prepare(self):
            pass

        def instance_ids(self):
            return ["only"]

        def load(self, instance_id):
            return LoadedTask(
                spec=TaskSpec(id=instance_id, prompt="p"),
                verify=lambda w, t: [], safety=None, gold=None,
            )

    assert isinstance(Minimal(), Benchmark)
    assert [t.id for t in load_tasks(Minimal())] == ["only"]


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


IDS = [f"i{n}" for n in range(50)]


def test_a_limit_samples_rather_than_taking_a_prefix():
    """Benchmarks are ordered by something — difficulty, repository, date — so
    the first N is a biased subset that still reads like a whole-benchmark
    score."""
    assert select(IDS, limit=10) != IDS[:10]


def test_sampling_is_reproducible_and_the_seed_moves_it():
    assert select(IDS, limit=10, seed=1) == select(IDS, limit=10, seed=1)
    assert select(IDS, limit=10, seed=1) != select(IDS, limit=10, seed=2)


def test_a_sample_keeps_the_benchmarks_own_order():
    """Instances are reported in benchmark order whatever was sampled, so two
    runs of the same subset line up when read side by side."""
    picked = select(IDS, limit=10, seed=3)
    assert picked == sorted(picked, key=IDS.index)


def test_a_limit_larger_than_the_benchmark_is_not_an_error():
    assert select(IDS, limit=500) == IDS


def test_selecting_named_instances():
    assert select(IDS, only=["i4", "i2"]) == ["i2", "i4"]


def test_an_unknown_instance_is_named_not_silently_dropped():
    with pytest.raises(BenchmarkError, match="i999"):
        select(IDS, only=["i999"])


def test_many_unknown_instances_are_summarised():
    with pytest.raises(BenchmarkError, match="and 5 more"):
        select(IDS, only=[f"x{n}" for n in range(10)])


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


def test_the_builtin_benchmarks_are_registered():
    assert {"local", "humaneval"} <= set(registered())


def test_resolving_by_name():
    assert resolve("local").name == "local"


def test_the_suffix_is_the_benchmarks_own_argument(tmp_path):
    assert resolve(f"local:{tmp_path}").root == tmp_path


def test_an_unknown_benchmark_lists_what_there_is():
    with pytest.raises(BenchmarkError, match="humaneval"):
        resolve("swe-bench-verified")


def test_a_third_party_benchmark_can_register_itself():
    """Nothing in agenteval needs to change to add one — the point of the
    protocol."""
    register("fake-for-test", lambda arg: LocalBenchmark(), "a test double")
    assert resolve("fake-for-test").name == "local"
    base._BENCHMARKS.pop("fake-for-test")


def test_the_summary_records_which_benchmark_and_how_big():
    """Recorded with the results because "20 of HumanEval's 164" is a different
    claim from "HumanEval", and the difference has to survive into the file."""
    assert summarise(LocalBenchmark()) == {
        "name": "local",
        "instances": len(list(DEFAULT_TASK_ROOT.glob("*/task.yaml"))),
    }


def test_downloads_are_cached_outside_the_repository():
    """This repository is public, and a benchmark file committed to GitHub is a
    benchmark file in the next model's training set."""
    assert "agenteval/tasks" not in str(cache_root())
    assert cache_root().is_absolute()


# --------------------------------------------------------------------------- #
# The local task directory, as a benchmark
# --------------------------------------------------------------------------- #


def test_the_existing_suite_goes_through_the_protocol_unchanged():
    """The abstraction was invented for downloaded benchmarks; if the suite we
    already had could not be expressed through it, the shape would be wrong."""
    tasks = load_tasks(LocalBenchmark())
    assert {"expense_approval", "revenue_reconciliation"} <= {t.id for t in tasks}
    assert all(t.verify for t in tasks)


def test_a_loaded_task_records_where_it_came_from():
    task = LocalBenchmark().load("ticket_triage")
    assert task.benchmark == "local"
    assert task.manifest()["benchmark"] == "local"


def test_instances_are_keyed_on_the_task_id_not_the_directory(tmp_path):
    """The two are allowed to differ, and --task selects on the id results are
    filed under."""
    directory = tmp_path / "some-folder"
    directory.mkdir()
    (directory / "task.yaml").write_text("id: the_real_id\nprompt: do a thing\n")
    (directory / "verify.py").write_text("def verify(world, trajectory): return []")
    benchmark = LocalBenchmark(tmp_path)
    assert benchmark.instance_ids() == ["the_real_id"]
    assert benchmark.load("the_real_id").id == "the_real_id"


def test_a_missing_task_directory_says_so(tmp_path):
    with pytest.raises(BenchmarkError, match="does not exist"):
        LocalBenchmark(tmp_path / "nope").prepare()


# --------------------------------------------------------------------------- #
# HumanEval, without the network
# --------------------------------------------------------------------------- #


PROBLEM = {
    "task_id": "HumanEval/0",
    "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Add.\"\"\"\n",
    "canonical_solution": "    return a + b\n",
    "entry_point": "add",
    "test": "def check(candidate):\n    assert candidate(2, 2) == 4\n",
}


@pytest.fixture
def humaneval(tmp_path):
    """A two-line dataset in the cache, so nothing here touches the network."""
    benchmark = HumanEvalBenchmark()
    benchmark.path = tmp_path / "HumanEval.jsonl.gz"
    benchmark.path.parent.mkdir(parents=True, exist_ok=True)
    second = {**PROBLEM, "task_id": "HumanEval/1"}
    with gzip.open(benchmark.path, "wt") as handle:
        handle.write(json.dumps(PROBLEM) + "\n" + json.dumps(second) + "\n")
    return benchmark


def test_the_dataset_is_read_from_the_cache(humaneval):
    assert humaneval.instance_ids() == ["HumanEval/0", "HumanEval/1"]


def test_preparing_twice_does_not_reread(humaneval):
    humaneval.prepare()
    humaneval.path.unlink()
    humaneval.prepare()  # idempotent, as the protocol requires
    assert humaneval.instance_ids()


def test_the_prompt_carries_the_stub_and_the_path(humaneval):
    spec = humaneval.load("HumanEval/0").spec
    assert "def add(a: int, b: int)" in spec.prompt
    assert "/workspace/solution.py" in spec.prompt


def test_nothing_is_seeded_into_the_container(humaneval):
    """The tests must not be reachable before the agent has finished. An agent
    that can read check() is being measured on something else."""
    environment = humaneval.load("HumanEval/0").spec.environment
    assert not environment.get("files")
    assert environment["network"] == "none"


def test_only_the_execution_tools_are_offered(humaneval):
    """A HumanEval instance has no CRM to search. Offering the simulated tools
    would spend the step budget teaching it what is irrelevant."""
    from agenteval.exec import EXEC_TOOLS

    allowed = humaneval.load("HumanEval/0").spec.allowed_tools
    assert allowed == list(EXEC_TOOLS)
    assert not [t for t in allowed if not t.startswith("exec_")]


def test_grading_needs_the_live_container(humaneval):
    """Nothing is left for a post-teardown verifier, because the verdict is an
    exit code rather than a file."""
    task = humaneval.load("HumanEval/0")
    assert task.grade_in_environment is not None
    assert task.verify(World({}), Trajectory("t", "a")) == []


def test_the_reference_solution_ships_as_a_replayable_trajectory(humaneval):
    """Every adapter should have one: it proves the container, the file path
    and the grader before a single token is paid for."""
    gold = humaneval.load("HumanEval/0").gold
    written = next(s for s in gold if s.get("tool") == "exec_write_file")
    assert "return a + b" in written["input"]["content"]


def test_the_image_is_the_benchmarks_argument():
    assert resolve("humaneval:swebench/instance:latest").image == (
        "swebench/instance:latest"
    )


def test_a_corrupt_cache_is_removed_rather_than_failing_forever(tmp_path):
    """An interrupted download would otherwise break every subsequent run with
    a message about gzip."""
    benchmark = HumanEvalBenchmark()
    benchmark.path = tmp_path / "HumanEval.jsonl.gz"
    benchmark.path.write_bytes(b"not gzip at all")
    with pytest.raises(BenchmarkError, match="removed"):
        benchmark.prepare()
    assert not benchmark.path.exists()


def test_a_failed_download_says_where_to_put_the_file_by_hand(tmp_path, monkeypatch):
    benchmark = HumanEvalBenchmark()
    benchmark.path = tmp_path / "HumanEval.jsonl.gz"

    def refuse(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(BenchmarkError, match="by hand"):
        benchmark.prepare()


def test_a_download_is_moved_into_place_whole(tmp_path, monkeypatch):
    """Written to a .part and renamed, so an interrupted fetch cannot leave a
    truncated file that looks cached."""
    benchmark = HumanEvalBenchmark()
    benchmark.path = tmp_path / "HumanEval.jsonl.gz"
    payload = gzip.compress((json.dumps(PROBLEM) + "\n").encode())

    class Response:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response())
    benchmark.prepare()
    assert benchmark.instance_ids() == ["HumanEval/0"]
    assert not list(tmp_path.glob("*.part"))


def test_asking_for_an_instance_that_is_not_there(humaneval):
    with pytest.raises(BenchmarkError, match="no such instance"):
        humaneval.load("HumanEval/9999")


# --------------------------------------------------------------------------- #
# The runner hook
# --------------------------------------------------------------------------- #


class FakeEnvironment:
    """Stands in for a container so the wiring can be tested without Docker."""

    def __init__(self, spec, docker="docker"):
        self.spec = spec
        self.container = "fake"
        self.stopped = False
        self.order: list[str] = []

    def start(self):
        pass

    def stop(self):
        self.stopped = True
        self.order.append("stop")

    def harvest(self):
        self.order.append("harvest")
        return {}

    def snapshot(self):
        return {"image": self.spec.image, "network": self.spec.network,
                "commands": 0, "log": []}


@pytest.fixture
def fake_container(monkeypatch):
    made = []

    def build(spec, *args, **kwargs):
        environment = FakeEnvironment(spec)
        made.append(environment)
        return environment

    monkeypatch.setattr("agenteval.runner.Environment", build)
    return made


def _task(grade_in_environment=None, verify=None):
    return LoadedTask(
        spec=TaskSpec(id="t", prompt="p", environment={"image": IMAGE}),
        verify=verify or (lambda w, t: [Check("from the world", passed=True)]),
        safety=None, gold=None, grade_in_environment=grade_in_environment,
    )


def test_in_container_checks_are_graded_before_teardown(fake_container):
    """The order that matters: a benchmark decides pass or fail by running the
    repository's own tests, and a destroyed container has no exit codes left."""
    seen = {}

    def grade(world, trajectory, environment):
        seen["alive"] = not environment.stopped
        environment.order.append("grade")
        return [Check("tests pass", passed=True)]

    result = asyncio.run(run_one(_task(grade), ScriptedAgent([]), RunConfig()))
    assert seen["alive"]
    assert fake_container[0].order == ["grade", "harvest", "stop"]
    assert result.status == "ok"


def test_in_container_checks_join_the_state_score(fake_container):
    def grade(world, trajectory, environment):
        return [Check("tests pass", passed=False, weight=3.0)]

    result = asyncio.run(run_one(_task(grade), ScriptedAgent([]), RunConfig()))
    names = [c.name for c in result.score.state_checks]
    assert names == ["tests pass", "from the world"]
    assert result.score.state_score == pytest.approx(0.25)


def test_a_broken_in_container_grader_is_a_harness_error(fake_container):
    """Not a score of zero: the agent may have solved it perfectly and the
    difference has to stay visible."""

    def grade(world, trajectory, environment):
        raise RuntimeError("docker exec died")

    result = asyncio.run(run_one(_task(grade), ScriptedAgent([]), RunConfig()))
    assert result.status == "harness_error"
    assert "docker exec died" in result.trajectory.error
    assert fake_container[0].stopped  # still cleaned up


def test_a_broken_verifier_does_not_discard_the_container_checks(fake_container):
    """They were the expensive ones, and they are already computed."""

    def grade(world, trajectory, environment):
        return [Check("tests pass", passed=True)]

    def verify(world, trajectory):
        raise ValueError("bad verifier")

    result = asyncio.run(
        run_one(_task(grade, verify), ScriptedAgent([]), RunConfig())
    )
    assert result.status == "harness_error"
    assert [c.name for c in result.score.state_checks] == ["tests pass"]


def test_a_task_without_the_hook_is_unaffected(fake_container):
    result = asyncio.run(run_one(_task(), ScriptedAgent([]), RunConfig()))
    assert [c.name for c in result.score.state_checks] == ["from the world"]


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


needs_image = pytest.mark.skipif(
    not (env_mod.available() and env_mod.image_present(IMAGE)),
    reason=f"{IMAGE} not built (docker build -f Dockerfile.exec .)",
)


def _run(task, script):
    return asyncio.run(run_one(task, ScriptedAgent(script), RunConfig()))


def _write(content, path="/workspace/solution.py"):
    return [{"tool": "exec_write_file", "input": {"path": path, "content": content}}]


@needs_image
def test_a_downloaded_benchmark_grades_its_own_reference_solution(humaneval):
    task = humaneval.load("HumanEval/0")
    result = _run(task, task.gold)
    assert result.score.overall == 1.0
    assert result.status == "ok"


@needs_image
@pytest.mark.parametrize(
    "label, script, detail",
    [
        ("wrong answer", _write("def add(a, b):\n    return 0\n"), "AssertionError"),
        ("no solution", [{"say": "done"}], "nothing at"),
        ("wrong path", _write("x = 1\n", "/workspace/answer.py"), "nothing at"),
        ("syntax error", _write("def add(:\n"), "SyntaxError"),
    ],
)
def test_wrong_answers_score_zero_for_the_right_reason(
    humaneval, label, script, detail
):
    """The check that a passing gold run cannot make: a grader that always says
    yes looks identical to a correct one until something fails."""
    result = _run(humaneval.load("HumanEval/0"), script)
    assert result.score.overall == 0.0
    assert detail in result.score.state_checks[0].detail


@needs_image
def test_a_solution_that_hangs_fails_rather_than_wedging_the_suite(humaneval):
    humaneval.timeout = 5.0
    result = _run(humaneval.load("HumanEval/0"), _write("def add(a, b):\n    while 1: pass\n"))
    assert result.score.overall == 0.0
    assert "did not finish" in result.score.state_checks[0].detail


@needs_image
def test_the_agent_cannot_read_the_tests_during_the_run(humaneval):
    """The reason grading is a hook rather than a seeded file."""
    result = _run(
        humaneval.load("HumanEval/0"),
        [{"tool": "exec_bash", "input": {"command": "cat /workspace/*.py 2>&1"}}],
    )
    assert "candidate(2, 2)" not in result.trajectory.calls[0].output


@needs_image
def test_the_solution_is_still_collected_for_review(humaneval):
    """Grading happens in the container; reading what the agent wrote afterwards
    still goes through `collect`, so the report has something to show."""
    result = _run(humaneval.load("HumanEval/0"), _write("def add(a, b):\n    return 0\n"))
    assert result.trajectory.environment["image"] == IMAGE
