"""Running task code in a container.

Split in two. The protocol and the container flags are tested without Docker,
because they are the security contract and must be checked on every machine.
The end-to-end tests need a built image and skip cleanly when there isn't one.
"""

import io
import json
import shutil
import subprocess

import pytest

from agenteval import Trajectory, World, _sandbox_entry
from agenteval.sandbox import HARDENING, Sandbox, SandboxedGrader, SandboxError
from agenteval.tasks import DEFAULT_TASK_ROOT, discover, load_task

VERIFY_SOURCE = '''
from agenteval import checks

def verify(world, trajectory):
    c = checks()
    c.add("ticket closed",
          world.find("tickets", "TKT-1")["status"] == "closed",
          detail="status was not closed")
    c.add("decided once", len(world.mutations_for("tickets", "update")) == 1)
    c.add("agent spoke", bool(trajectory.final_text))
    return c.done()

def safety(world, trajectory):
    return ["a violation"] if world.outbox else []

GOLD = [{"say": "done"}]
'''

SEED = {
    "tickets": [
        {"id": "TKT-1", "subject": "s", "body": "b", "status": "open",
         "priority": "P3", "team": "support", "comments": []}
    ]
}


def make_world_and_trajectory():
    from agenteval.world import tickets

    world = World(SEED)
    tickets.update(world, "TKT-1", status="closed")
    trajectory = Trajectory(task_id="t", agent="a")
    trajectory.final_text = "done"
    return world, trajectory


# --------------------------------------------------------------------------- #
# The container flags are the boundary
# --------------------------------------------------------------------------- #


def test_the_sandbox_has_no_network():
    """The single most important flag: with no route out, a malicious verifier
    has nowhere to send anything it manages to read."""
    assert HARDENING[HARDENING.index("--network") + 1] == "none"


def test_the_sandbox_inherits_no_environment():
    """Why only task code is containerised and not the whole run: the API key
    lives in the agent's environment, and the sandbox must not see it."""
    assert HARDENING[HARDENING.index("--env-file") + 1] == "/dev/null"


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--user", "65534:65534"),      # nobody
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--pids-limit", "256"),        # fork bombs
        ("--memory", "512m"),
    ],
)
def test_remaining_hardening_flags(flag, value):
    assert HARDENING[HARDENING.index(flag) + 1] == value


def test_the_root_filesystem_is_read_only():
    assert "--read-only" in HARDENING
    tmpfs = HARDENING[HARDENING.index("--tmpfs") + 1]
    assert "noexec" in tmpfs and "nosuid" in tmpfs


# --------------------------------------------------------------------------- #
# The in-container entrypoint, tested directly
# --------------------------------------------------------------------------- #


def test_load_returns_gold_without_the_caller_executing_anything():
    out = _sandbox_entry._handle(
        {"op": "load", "task_id": "t", "verify_source": VERIFY_SOURCE}
    )
    assert out == {"ok": True, "gold": [{"say": "done"}], "has_safety": True}


def test_load_rejects_a_module_with_no_verify():
    out = _sandbox_entry._handle(
        {"op": "load", "task_id": "t", "verify_source": "x = 1"}
    )
    assert out["ok"] is False and "no verify()" in out["error"]


def test_grade_returns_checks_and_violations_from_one_crossing():
    world, trajectory = make_world_and_trajectory()
    out = _sandbox_entry._handle({
        "op": "grade", "task_id": "t", "verify_source": VERIFY_SOURCE,
        "world": world.to_dict(), "trajectory": trajectory.to_dict(),
    })
    assert out["ok"] is True
    assert [c["name"] for c in out["checks"]] == [
        "ticket closed", "decided once", "agent spoke"
    ]
    assert all(c["passed"] for c in out["checks"])
    assert out["violations"] == []


def test_the_mutation_log_survives_the_crossing():
    """Checks like "decided exactly once" read the log, not the end state, so
    serialising only the records would silently change what they measure."""
    world, trajectory = make_world_and_trajectory()
    out = _sandbox_entry._handle({
        "op": "grade", "task_id": "t", "verify_source": VERIFY_SOURCE,
        "world": world.to_dict(), "trajectory": trajectory.to_dict(),
    })
    assert next(c for c in out["checks"] if c["name"] == "decided once")["passed"]


def run_entry(request, monkeypatch, capsys):
    """Drive the entrypoint the way the container does — stdin to stdout."""
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(request)))
    assert _sandbox_entry.main() == 0
    return json.loads(capsys.readouterr().out)


def test_task_code_that_raises_is_reported_not_propagated(monkeypatch, capsys):
    """`main` is the containment boundary, not `_handle`: a task that blows up
    should fail its own run and leave the rest of the suite standing."""
    out = run_entry(
        {"op": "load", "task_id": "t",
         "verify_source": "raise RuntimeError('bad task')"},
        monkeypatch, capsys,
    )
    assert out["ok"] is False
    assert "bad task" in out["error"]


def test_a_verifier_that_raises_mid_grade_is_reported(monkeypatch, capsys):
    world, trajectory = make_world_and_trajectory()
    out = run_entry({
        "op": "grade", "task_id": "t",
        "verify_source": "def verify(w, t):\n    raise KeyError('missing')\n",
        "world": world.to_dict(), "trajectory": trajectory.to_dict(),
    }, monkeypatch, capsys)
    assert out["ok"] is False and "KeyError" in out["error"]


def test_an_unknown_op_is_refused(monkeypatch, capsys):
    out = run_entry(
        {"op": "sudo", "task_id": "t", "verify_source": VERIFY_SOURCE},
        monkeypatch, capsys,
    )
    assert out["ok"] is False and "unknown op" in out["error"]


def test_main_never_raises_on_malformed_input(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert _sandbox_entry.main() == 0
    assert json.loads(capsys.readouterr().out)["ok"] is False


# --------------------------------------------------------------------------- #
# Host side, with a stubbed docker
# --------------------------------------------------------------------------- #


class FakeDocker:
    def __init__(self, response=None, returncode=0, stdout=None, raises=None):
        self.calls = []
        self._response = response
        self._returncode = returncode
        self._stdout = stdout
        self._raises = raises

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self._raises:
            raise self._raises
        stdout = (self._stdout if self._stdout is not None
                  else json.dumps(self._response or {"ok": True}))
        return subprocess.CompletedProcess(cmd, self._returncode, stdout, "stderr")


def test_every_hardening_flag_reaches_the_docker_invocation(monkeypatch):
    fake = FakeDocker({"ok": True, "gold": None, "has_safety": False})
    monkeypatch.setattr(subprocess, "run", fake)
    Sandbox().load("t", VERIFY_SOURCE)

    [(cmd, kwargs)] = fake.calls
    assert cmd[:4] == ["docker", "run", "--rm", "-i"]
    for flag in HARDENING:
        assert flag in cmd
    assert cmd[-1] == "agenteval-sandbox:latest"
    # The task source goes in on stdin, never into the image or a mount.
    assert json.loads(kwargs["input"])["verify_source"] == VERIFY_SOURCE


def test_a_hanging_verifier_fails_its_own_run(monkeypatch):
    monkeypatch.setattr(subprocess, "run", FakeDocker(
        raises=subprocess.TimeoutExpired("docker", 60)))
    with pytest.raises(SandboxError, match="exceeded 60s"):
        Sandbox().load("t", VERIFY_SOURCE)


def test_a_nonzero_exit_is_surfaced(monkeypatch):
    monkeypatch.setattr(subprocess, "run", FakeDocker(returncode=137))
    with pytest.raises(SandboxError, match="exited 137"):
        Sandbox().load("t", VERIFY_SOURCE)


def test_garbage_on_stdout_is_surfaced(monkeypatch):
    monkeypatch.setattr(subprocess, "run", FakeDocker(stdout="<html>nope"))
    with pytest.raises(SandboxError, match="no JSON"):
        Sandbox().load("t", VERIFY_SOURCE)


def test_task_failure_becomes_a_sandbox_error(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        FakeDocker({"ok": False, "error": "boom in verify"}))
    with pytest.raises(SandboxError, match="boom in verify"):
        Sandbox().load("t", VERIFY_SOURCE)


def test_verify_and_safety_share_one_container(monkeypatch):
    """A container costs orders of magnitude more than the work inside it."""
    fake = FakeDocker({"ok": True, "checks": [], "violations": ["v"]})
    monkeypatch.setattr(subprocess, "run", fake)

    world, trajectory = make_world_and_trajectory()
    grader = SandboxedGrader(Sandbox(), "t", VERIFY_SOURCE)
    grader.verify(world, trajectory)
    grader.safety(world, trajectory)
    assert len(fake.calls) == 1


def test_a_different_run_is_not_served_from_cache(monkeypatch):
    fake = FakeDocker({"ok": True, "checks": [], "violations": []})
    monkeypatch.setattr(subprocess, "run", fake)

    grader = SandboxedGrader(Sandbox(), "t", VERIFY_SOURCE)
    grader.verify(*make_world_and_trajectory())
    grader.verify(*make_world_and_trajectory())
    assert len(fake.calls) == 2


def test_preflight_explains_a_missing_docker(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(SandboxError, match="not on PATH"):
        Sandbox().preflight()


def test_preflight_explains_a_missing_image(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(subprocess, "run", FakeDocker(returncode=1))
    with pytest.raises(SandboxError, match="sandbox build"):
        Sandbox().preflight()


def test_loading_a_task_with_a_sandbox_never_imports_it(monkeypatch, tmp_path):
    """The whole point of the flag: untrusted code must not touch this
    interpreter, so the in-process import path must not run at all."""
    directory = tmp_path / "t"
    directory.mkdir()
    (directory / "task.yaml").write_text("id: t\nprompt: p\n")
    (directory / "verify.py").write_text(VERIFY_SOURCE)

    import agenteval.tasks as tasks_mod

    def explode(*a, **k):
        raise AssertionError("verify.py was imported on the host")

    monkeypatch.setattr(tasks_mod, "_load_module", explode)
    monkeypatch.setattr(subprocess, "run", FakeDocker(
        {"ok": True, "gold": [{"say": "done"}], "has_safety": True}))

    task = load_task(directory, sandbox=Sandbox())
    assert task.gold == [{"say": "done"}]
    assert task.safety is not None


# --------------------------------------------------------------------------- #
# End to end, when an image is available
# --------------------------------------------------------------------------- #


def _image_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(
            ["docker", "image", "inspect", "agenteval-sandbox:latest"],
            capture_output=True, timeout=30,
        ).returncode == 0
    except Exception:  # noqa: BLE001
        return False


needs_image = pytest.mark.skipif(
    not _image_ready(),
    reason="sandbox image not built (`agenteval sandbox build`)",
)


@needs_image
@pytest.mark.parametrize(
    "task", discover(DEFAULT_TASK_ROOT), ids=lambda t: t.id
)
def test_sandboxed_grading_agrees_with_in_process(task):
    """Sandboxing is a deployment choice, so it must not change any score."""
    import asyncio

    from agenteval import RunConfig, ScriptedAgent, run_one

    sandboxed = load_task(
        DEFAULT_TASK_ROOT / task.id, sandbox=Sandbox()
    )
    assert sandboxed.gold == task.gold

    def score(loaded):
        return asyncio.run(
            run_one(loaded, ScriptedAgent(loaded.gold, "gold"), RunConfig())
        ).score

    plain, boxed = score(task), score(sandboxed)
    assert [(c.name, c.passed, c.weight) for c in boxed.state_checks] == \
           [(c.name, c.passed, c.weight) for c in plain.state_checks]
    assert boxed.safety_violations == plain.safety_violations
    assert boxed.overall == plain.overall == 1.0


@needs_image
def test_a_hostile_verifier_is_contained(tmp_path, monkeypatch):
    """The claim this feature exists to make, exercised against real Docker."""
    directory = tmp_path / "evil"
    directory.mkdir()
    (directory / "task.yaml").write_text("id: evil\nprompt: p\n")
    (directory / "verify.py").write_text(
        "import os, socket\n"
        "from agenteval import checks\n"
        "def verify(world, trajectory):\n"
        "    c = checks()\n"
        "    c.add('secret=' + (os.environ.get('FAKE_SECRET') or 'none'), True)\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "        net = 'reachable'\n"
        "    except Exception:\n"
        "        net = 'blocked'\n"
        "    c.add('net=' + net, True)\n"
        "    return c.done()\n"
        "GOLD = [{'say': 'x'}]\n"
    )
    monkeypatch.setenv("FAKE_SECRET", "sk-ant-PRETEND")

    task = load_task(directory, sandbox=Sandbox())
    world, trajectory = make_world_and_trajectory()
    names = [c.name for c in task.verify(world, trajectory)]

    assert "secret=none" in names, "the host credential reached the sandbox"
    assert "net=blocked" in names, "the sandbox reached the network"


# --------------------------------------------------------------------------- #
# Image build
# --------------------------------------------------------------------------- #


def test_build_invokes_docker_with_the_repo_dockerfile(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(subprocess, "run", fake)
    Sandbox().build()

    [(cmd, _)] = fake.calls
    assert cmd[:4] == ["docker", "build", "-t", "agenteval-sandbox:latest"]
    assert cmd[cmd.index("-f") + 1].endswith("Dockerfile")


def test_a_failed_build_surfaces_dockers_own_output(monkeypatch):
    def failing(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "step 3/5 failed: no space")

    monkeypatch.setattr(subprocess, "run", failing)
    with pytest.raises(SandboxError, match="no space"):
        Sandbox().build()


def test_build_reports_a_missing_dockerfile(monkeypatch, tmp_path):
    import agenteval.sandbox as sandbox_mod

    monkeypatch.setattr(sandbox_mod, "DOCKERFILE", tmp_path / "absent")
    with pytest.raises(SandboxError, match="no Dockerfile"):
        Sandbox().build()
