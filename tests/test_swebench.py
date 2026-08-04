"""SWE-bench: real GitHub issues, graded by each repository's own tests.

Almost everything here runs against a fixture instance rather than the network
or a 4 GB image. What is being tested is the adapter — the container it asks
for, the patch it captures, and how upstream's report becomes our checks. The
eval scripts and log parsers belong to the `swebench` package and are its
business, not ours; the one thing worth asserting about them is that we call
them rather than reimplementing them.
"""

import gzip
import json

import pytest

from agenteval import RunConfig, ScriptedAgent, World, run_one
from agenteval.benchmarks import BenchmarkError, SWEBenchBenchmark, resolve
from agenteval.benchmarks import swebench as mod
from agenteval.exec import environment as env_mod
from agenteval.types import Trajectory

#: The adapter is useless without upstream's eval scripts, and that is the
#: point rather than an oversight — see the module docstring in swebench.py.
pytest.importorskip("swebench", reason="pip install 'agenteval[swebench]'")

INSTANCE = {
    "repo": "astropy/astropy",
    "instance_id": "astropy__astropy-12907",
    "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
    "patch": "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n",
    "test_patch": "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n",
    "problem_statement": "separability_matrix computes the wrong thing",
    "hints_text": "",
    "created_at": "2022-03-03T15:14:54Z",
    "version": "4.3",
    "FAIL_TO_PASS": '["t.py::test_a", "t.py::test_b"]',
    "PASS_TO_PASS": '["t.py::test_c"]',
    "environment_setup_commit": "298ccb478e6bf092953bca67a3d29dc6c35f6752",
}


@pytest.fixture
def benchmark(tmp_path):
    """A one-instance dataset in the cache, so nothing here hits the network."""
    b = SWEBenchBenchmark("lite")
    b.path = tmp_path / "swebench.jsonl.gz"
    b.path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(b.path, "wt") as handle:
        handle.write(json.dumps(INSTANCE))
    return b


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #


def test_the_split_is_the_benchmarks_argument():
    assert resolve("swebench:verified").dataset == "princeton-nlp/SWE-bench_Verified"
    assert resolve("swebench").dataset == "princeton-nlp/SWE-bench_Lite"


def test_an_unrecognised_split_is_passed_through_rather_than_rejected():
    """A fork or a private variant should work without a code change here."""
    assert resolve("swebench:my-org/SWE-bench-fork").dataset == (
        "my-org/SWE-bench-fork"
    )


def test_the_split_reaches_the_results(benchmark):
    """`swebench:lite` and `swebench:verified` are different claims and the
    saved record has to tell them apart."""
    assert benchmark.name == "swebench:lite"
    assert benchmark.load(INSTANCE["instance_id"]).benchmark == "swebench:lite"


def test_instances_keep_their_own_ids(benchmark):
    assert benchmark.instance_ids() == ["astropy__astropy-12907"]


def test_asking_for_an_instance_that_is_not_there(benchmark):
    with pytest.raises(BenchmarkError, match="no such instance"):
        benchmark.load("django__django-99999")


def test_a_corrupt_cache_is_removed_rather_than_failing_forever(benchmark):
    benchmark.path.write_bytes(b"not gzip")
    benchmark._instances = {}
    with pytest.raises(BenchmarkError, match="removed"):
        benchmark.prepare()
    assert not benchmark.path.exists()


def test_pagination_walks_the_whole_split(monkeypatch, tmp_path):
    """SWE-bench Lite is 300 rows and the datasets server pages at 100, so a
    single request would silently produce a third of a benchmark."""
    b = SWEBenchBenchmark("lite")
    b.path = tmp_path / "swebench.jsonl.gz"
    served = [
        {**INSTANCE, "instance_id": f"repo__project-{n}"} for n in range(250)
    ]

    class Reply:
        def __init__(self, offset):
            self.offset = offset

        def read(self):
            page = served[self.offset:self.offset + mod.PAGE]
            return json.dumps(
                {"num_rows_total": len(served),
                 "rows": [{"row": r} for r in page]}
            ).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    def fake_open(url, **kwargs):
        offset = int(url.split("offset=")[1].split("&")[0])
        return Reply(offset)

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    b.prepare()
    assert len(b.instance_ids()) == 250


def test_an_unreachable_datasets_server_names_the_split(monkeypatch, tmp_path):
    b = SWEBenchBenchmark("lite")
    b.path = tmp_path / "x.jsonl.gz"

    def refuse(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(BenchmarkError, match="SWE-bench_Lite"):
        b.prepare()


def test_the_datasets_server_refusing_is_reported_not_swallowed(
    monkeypatch, tmp_path
):
    b = SWEBenchBenchmark("nope/does-not-exist")
    b.path = tmp_path / "x.jsonl.gz"

    class Reply:
        def read(self):
            return json.dumps({"error": "Not found."}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Reply())
    with pytest.raises(BenchmarkError, match="refused"):
        b.prepare()


# --------------------------------------------------------------------------- #
# The container an instance asks for
# --------------------------------------------------------------------------- #


def test_the_image_is_this_instances_own(benchmark):
    """The seam. One image per instance is the difference between supporting
    SWE-bench and rewriting the harness for it."""
    environment = benchmark.load(INSTANCE["instance_id"]).spec.environment
    assert environment["image"] == (
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    )


def test_the_architecture_is_pinned_not_detected(benchmark):
    """Upstream defaults arch from the host. On Apple silicon that yields an
    arm64 image name nobody has ever published, and the failure arrives as a
    404 half a minute into a paid run."""
    environment = benchmark.load(INSTANCE["instance_id"]).spec.environment
    assert "x86_64" in environment["image"]
    assert environment["platform"] == "linux/amd64"


def test_the_repository_has_to_be_writable(benchmark):
    """Every other task here runs on a read-only root. These images assume
    root, a writable /testbed and a conda environment they install into."""
    environment = benchmark.load(INSTANCE["instance_id"]).spec.environment
    assert environment["read_only_root"] is False
    assert environment["user"] == "root"
    assert environment["workdir"] == "/testbed"


def test_egress_is_allowlisted_rather_than_opened(benchmark):
    """Most eval scripts `pip install -e .`, which is a real reason to need the
    network and not a reason to hand over an unfiltered one."""
    environment = benchmark.load(INSTANCE["instance_id"]).spec.environment
    assert "pypi.org" in environment["allow_hosts"]
    assert "network" not in environment  # the gateway replaces it


def test_a_run_says_what_it_is_about_to_download(benchmark, monkeypatch, capsys):
    """Three hundred pulls of about a gigabyte each is a different proposition
    from a shared image, and a full disk two hours in is a bad way to learn
    that."""
    from agenteval import cli

    tasks = [
        benchmark.load(INSTANCE["instance_id"]),
        benchmark.load(INSTANCE["instance_id"]),
    ]
    tasks[1].spec.environment = {**tasks[1].spec.environment, "image": "other:latest"}
    monkeypatch.setattr(env_mod, "available", lambda *a, **k: True)
    monkeypatch.setattr(env_mod, "image_present", lambda *a, **k: False)
    cli.report_images_to_pull(tasks)
    out = capsys.readouterr().out
    assert "2 of 2 container images are not local" in out
    assert "--limit" in out


def test_nothing_is_said_when_the_images_are_already_there(
    benchmark, monkeypatch, capsys
):
    from agenteval import cli

    tasks = [benchmark.load(INSTANCE["instance_id"])] * 2
    monkeypatch.setattr(env_mod, "available", lambda *a, **k: True)
    monkeypatch.setattr(env_mod, "image_present", lambda *a, **k: True)
    cli.report_images_to_pull(tasks)
    assert capsys.readouterr().out == ""


def test_a_shared_image_benchmark_says_nothing(monkeypatch, capsys):
    """One image is a one-off cost anyone building this repo has already paid."""
    from agenteval import cli
    from agenteval.benchmarks import HumanEvalBenchmark

    monkeypatch.setattr(env_mod, "available", lambda *a, **k: True)
    monkeypatch.setattr(env_mod, "image_present", lambda *a, **k: False)
    b = HumanEvalBenchmark()
    b._problems = {"HumanEval/0": {
        "task_id": "HumanEval/0", "prompt": "def f():\n", "entry_point": "f",
        "canonical_solution": "    pass\n", "test": "def check(c):\n    pass\n",
    }}
    cli.report_images_to_pull([b.load("HumanEval/0")])
    assert capsys.readouterr().out == ""


def test_a_platform_pin_reaches_docker():
    spec = env_mod.EnvironmentSpec(platform="linux/amd64")
    args = env_mod.Environment(spec)._run_args()
    assert args[args.index("--platform") + 1] == "linux/amd64"


def test_no_platform_pin_leaves_docker_to_choose():
    assert "--platform" not in env_mod.Environment(env_mod.EnvironmentSpec())._run_args()


def test_the_prompt_tells_it_not_to_touch_the_tests(benchmark):
    """It could not cheat that way anyway — the eval script checks test files
    back out first — but an agent that spends ten steps editing tests that get
    reverted has been set up to fail rather than measured."""
    prompt = benchmark.load(INSTANCE["instance_id"]).spec.prompt
    assert "Do not modify or add tests" in prompt
    assert INSTANCE["problem_statement"] in prompt


def test_the_reference_pull_request_ships_as_gold(benchmark):
    """For a benchmark this expensive it earns its keep: it exercises the
    image, the platform pin, the allowlist, the eval script and upstream's
    grader for the price of one container."""
    gold = benchmark.load(INSTANCE["instance_id"]).gold
    written = next(s for s in gold if s.get("tool") == "exec_write_file")
    assert written["input"]["content"] == INSTANCE["patch"]


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


class FakeEnvironment:
    """A container that answers with whatever the test wants."""

    def __init__(self, diff="", exit_code=0, timed_out=False):
        self.spec = env_mod.EnvironmentSpec()
        self.container = "fake"
        self.commands = []
        self._diff = diff
        self._exit_code = exit_code
        self._timed_out = timed_out

    def exec(self, command, timeout=None):
        self.commands.append(command)
        if "git -c core.fileMode=false diff" in command:
            return env_mod.ExecResult(0, self._diff, "")
        return env_mod.ExecResult(
            self._exit_code, "test output", "", timed_out=self._timed_out
        )

    def write_file(self, path, content):
        self.commands.append(f"write {path}")
        return env_mod.ExecResult(0, "", "")


PATCH = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"


@pytest.fixture
def grading_returns(monkeypatch):
    """Replace only upstream's report, keeping its real eval-script builder.

    Captured before patching: a replacement that reaches back through the name
    it is replacing calls itself.
    """
    real_make_test_spec, _ = mod._require_swebench()

    def use(report):
        monkeypatch.setattr(
            mod, "_require_swebench",
            lambda: (real_make_test_spec, lambda **kw: report),
        )

    return use


def _grade(benchmark, environment):
    task = benchmark.load(INSTANCE["instance_id"])
    return task.grade_in_environment(
        World({}), Trajectory("t", "a"), environment
    )


def test_the_working_tree_is_the_answer_and_is_captured_first(benchmark):
    """The eval script checks test files back out, so a diff taken afterwards
    is a different diff."""
    environment = FakeEnvironment(diff=PATCH)
    world = World({})
    task = benchmark.load(INSTANCE["instance_id"])
    task.grade_in_environment(world, Trajectory("t", "a"), environment)
    assert environment.commands[0].startswith("cd /testbed && git")
    saved = world.find("documents", f"{INSTANCE['instance_id']}.patch")
    assert saved["content"] == PATCH


def test_an_untouched_repository_fails_without_running_anything(benchmark):
    """Reinstalling the project for several minutes to prove nothing changed
    is time spent learning what the diff already said."""
    environment = FakeEnvironment(diff="")
    checks = _grade(benchmark, environment)
    assert checks[0].name == "resolved" and not checks[0].passed
    assert "clean" in checks[0].detail
    assert not any("eval.sh" in c for c in environment.commands)


def test_a_suite_that_never_finishes_fails_the_instance(benchmark):
    checks = _grade(benchmark, FakeEnvironment(diff=PATCH, timed_out=True))
    assert not checks[0].passed and "did not finish" in checks[0].detail


def test_an_unreadable_log_is_a_failure_not_a_crash(benchmark, monkeypatch):
    """A benchmark of 300 instances will meet a log its parser does not
    recognise, and that must cost one instance rather than the run."""
    real_make_test_spec, _ = mod._require_swebench()

    def raising(**kwargs):
        raise ValueError("unrecognised runner")

    monkeypatch.setattr(
        mod, "_require_swebench", lambda: (real_make_test_spec, raising)
    )
    checks = _grade(benchmark, FakeEnvironment(diff=PATCH))
    assert not checks[0].passed and "could not read the test log" in checks[0].detail


def test_upstreams_report_becomes_our_checks(benchmark, grading_returns):
    report = {
        INSTANCE["instance_id"]: {
            "resolved": True,
            "tests_status": {
                "FAIL_TO_PASS": {"success": ["t::a", "t::b"], "failure": []},
                "PASS_TO_PASS": {"success": ["t::c"], "failure": []},
            },
        }
    }
    grading_returns(report)
    checks = _grade(benchmark, FakeEnvironment(diff=PATCH))
    assert checks[0].name == "resolved" and checks[0].passed
    assert [c.detail for c in checks[1:]] == ["2/2 passed", "1/1 passed"]


def test_only_resolved_carries_weight(benchmark, grading_returns):
    """So the suite mean is directly comparable to a published % resolved. The
    breakdown is the first thing anyone reads when a result surprises them, and
    folding it into the score would make the number unrecognisable."""
    report = {
        INSTANCE["instance_id"]: {
            "resolved": False,
            "tests_status": {
                "FAIL_TO_PASS": {"success": ["t::a"], "failure": ["t::b"]},
                "PASS_TO_PASS": {"success": ["t::c"], "failure": []},
            },
        }
    }
    grading_returns(report)
    checks = _grade(benchmark, FakeEnvironment(diff=PATCH))
    assert [c.weight for c in checks] == [1.0, 0.0, 0.0]
    from agenteval.types import Score

    assert Score(state_checks=checks).state_score == 0.0


def test_a_failure_says_which_half_of_the_criterion_broke(benchmark, grading_returns):
    """Not fixing the bug and breaking something else on the way out are
    different failures and want different next steps."""
    report = {
        INSTANCE["instance_id"]: {
            "resolved": False,
            "tests_status": {
                "FAIL_TO_PASS": {"success": [], "failure": ["t::a"]},
                "PASS_TO_PASS": {"success": [], "failure": ["t::c", "t::d"]},
            },
        }
    }
    grading_returns(report)
    detail = _grade(benchmark, FakeEnvironment(diff=PATCH))[0].detail
    assert "did not fix: t::a" in detail
    assert "regressed: t::c, t::d" in detail


def test_a_long_failure_list_is_summarised(benchmark, grading_returns):
    report = {
        INSTANCE["instance_id"]: {
            "resolved": False,
            "tests_status": {
                "FAIL_TO_PASS": {"success": [], "failure": [f"t::{n}" for n in range(9)]},
                "PASS_TO_PASS": {"success": [], "failure": []},
            },
        }
    }
    grading_returns(report)
    assert "and 6 more" in _grade(benchmark, FakeEnvironment(diff=PATCH))[0].detail


def test_the_eval_script_is_upstreams_not_ours(benchmark):
    """The one thing worth asserting about the script: that we did not write
    it. Which test command belongs to django 4.1 versus 3.2 is forty kilobytes
    of version-specific detail that upstream revises, and a copy would drift
    while still producing numbers that looked like SWE-bench."""
    environment = FakeEnvironment(diff=PATCH)
    _grade(benchmark, environment)
    staged = [c for c in environment.commands if c.startswith("write ")]
    assert staged == ["write /eval.sh"]
    test_spec = benchmark._spec_for(INSTANCE)
    assert "conda activate testbed" in test_spec.eval_script
    assert INSTANCE["test_patch"] in test_spec.eval_script


def test_a_script_that_cannot_be_staged_fails_the_instance(benchmark, monkeypatch):
    """One instance, not the run — a full disk in a container is that
    container's problem."""
    environment = FakeEnvironment(diff=PATCH)
    environment.write_file = lambda path, content: env_mod.ExecResult(
        1, "", "no space left on device"
    )
    checks = _grade(benchmark, environment)
    assert not checks[0].passed
    assert "could not stage the eval script" in checks[0].detail
