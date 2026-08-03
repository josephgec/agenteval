"""The execution stack: containers that tool calls run inside.

Split like the sandbox tests. Container flags and wiring are checked without
Docker, because they are the isolation contract. The end-to-end tests need the
exec image and skip cleanly without it.
"""

import asyncio
import subprocess

import pytest

from agenteval import RunConfig, ScriptedAgent, TaskSpec, Trajectory, World, run_one
from agenteval.exec import environment as env_mod
from agenteval.exec import EXEC_TOOLS, Environment, EnvironmentSpec, attach
from agenteval.exec.environment import EnvironmentError_, ExecResult
from agenteval.registry import REGISTRY, ToolSession
from agenteval.state import WorldError
from agenteval.tasks import DEFAULT_TASK_ROOT, discover
from agenteval.types import Check

IMAGE = env_mod.DEFAULT_IMAGE


# --------------------------------------------------------------------------- #
# The container flags are the isolation contract
# --------------------------------------------------------------------------- #


def args_for(**kwargs) -> list[str]:
    return Environment(EnvironmentSpec(**kwargs))._run_args()


def test_the_workspace_defaults_to_no_network():
    """Generated code is the most untrusted thing in the system. A task that
    genuinely needs egress opts in, and that choice is then visible in its
    task.yaml where it can be reviewed."""
    args = args_for()
    assert args[args.index("--network") + 1] == "none"


def test_no_host_environment_reaches_the_container():
    """The API key lives in the agent's environment on the host; generated code
    must not be able to read it."""
    args = args_for()
    assert args[args.index("--env-file") + 1] == "/dev/null"


@pytest.mark.parametrize(
    "flag", ["--read-only", "--security-opt", "--cap-drop", "--pids-limit"]
)
def test_remaining_hardening(flag):
    assert flag in args_for()


def test_the_workspace_is_writable_by_any_uid():
    """A tmpfs arrives owned by root, which a non-root image cannot write to —
    and pinning a uid here would break every benchmark image that chose a
    different one. Sticky world-writable, the /tmp convention, works for all."""
    args = args_for()
    workspace = next(a for a in args if a.startswith("/workspace:"))
    assert "mode=1777" in workspace
    assert "exec" in workspace  # code has to be runnable


def test_a_task_can_ask_for_network_and_a_bigger_workspace():
    args = args_for(network="bridge", workspace_mb=4096, memory="8g")
    assert args[args.index("--network") + 1] == "bridge"
    assert "size=4096m" in next(a for a in args if a.startswith("/workspace:"))
    assert args[args.index("--memory") + 1] == "8g"


def test_a_benchmark_image_can_run_as_its_own_user():
    assert "--user" not in args_for()                     # image's default
    assert args_for(user="root")[args_for(user="root").index("--user") + 1] == "root"


def test_the_image_is_a_parameter():
    """The seam that keeps a downloaded benchmark from needing a rewrite."""
    args = args_for(image="swebench/instance-1234:latest")
    assert "swebench/instance-1234:latest" in args


def test_unknown_environment_settings_are_rejected():
    with pytest.raises(EnvironmentError_, match="unknown environment settings"):
        EnvironmentSpec.from_config({"imagge": "typo"})


def test_no_environment_block_means_no_container():
    assert EnvironmentSpec.from_config(None) is None
    assert EnvironmentSpec.from_config({}) is None


# --------------------------------------------------------------------------- #
# Exposure
# --------------------------------------------------------------------------- #


def test_exec_tools_are_hidden_from_tasks_without_a_container():
    """Offering exec_bash and then refusing every call would just spend the
    agent's step budget teaching it what it cannot do."""
    spec = TaskSpec(id="t", prompt="p")
    session = ToolSession(World({}), spec, Trajectory("t", "a"))
    assert not [t for t in session.tools if t.name in EXEC_TOOLS]


def test_exec_tools_appear_when_a_task_declares_one():
    spec = TaskSpec(id="t", prompt="p", environment={"image": IMAGE})
    session = ToolSession(World({}), spec, Trajectory("t", "a"))
    exposed = {t.name for t in session.tools}
    assert set(EXEC_TOOLS) <= exposed
    # And they sit alongside the simulated tools, not instead of them.
    assert "tickets_create" in exposed


def test_exec_tools_are_registered_like_any_other():
    """They inherit the audit trail, the step budget and forbidden-tool
    blocking by being ordinary tools rather than a special case."""
    for name in EXEC_TOOLS:
        assert name in REGISTRY
        assert REGISTRY[name].schema["additionalProperties"] is False


def test_calling_an_exec_tool_without_a_container_says_why():
    spec = TaskSpec(id="t", prompt="p", allowed_tools=["exec_bash"])
    world = World({})
    session = ToolSession(world, spec, Trajectory("t", "a"))
    text, is_error = session.call("exec_bash", {"command": "ls"})
    assert is_error and "no execution environment" in text


# --------------------------------------------------------------------------- #
# Output shaping
# --------------------------------------------------------------------------- #


def test_a_timeout_is_reported_rather_than_raised():
    """A hung build should fail its own run, not wedge the suite."""
    assert "Timed out" in ExecResult(124, "partial", "", timed_out=True).render()


def test_stderr_and_exit_code_reach_the_agent():
    rendered = ExecResult(1, "some output", "boom").render()
    assert "some output" in rendered and "boom" in rendered and "exit 1" in rendered


def test_silent_success_still_says_something():
    assert ExecResult(0, "", "").render() == "(no output)"


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def _ready() -> bool:
    return env_mod.available() and env_mod.image_present(IMAGE)


needs_image = pytest.mark.skipif(
    not _ready(), reason=f"{IMAGE} not built (docker build -f Dockerfile.exec .)"
)


@needs_image
def test_a_container_runs_commands_and_keeps_state_between_them():
    """One container per run, not per call — a benchmark instance is a session
    with a repo, a virtualenv, and files written three steps ago."""
    with Environment(EnvironmentSpec(files={"/workspace/a.txt": "hello"})) as env:
        assert env.exec("cat /workspace/a.txt").stdout.strip() == "hello"
        env.exec("echo second > /workspace/b.txt")
        assert env.exec("cat /workspace/b.txt").stdout.strip() == "second"
        assert env.exec("python -c 'print(6*7)'").stdout.strip() == "42"


@needs_image
def test_generated_code_cannot_reach_the_network():
    with Environment(EnvironmentSpec()) as env:
        result = env.exec(
            "python -c \"import socket;"
            "socket.create_connection(('1.1.1.1',53),timeout=3)\" 2>&1"
        )
        assert not result.ok


@needs_image
def test_generated_code_cannot_read_the_hosts_credentials(monkeypatch):
    monkeypatch.setenv("FAKE_SECRET", "sk-ant-PRETEND")
    with Environment(EnvironmentSpec()) as env:
        assert "PRETEND" not in env.exec("env").stdout


@needs_image
def test_file_contents_survive_the_shell_verbatim():
    """Written on stdin rather than interpolated into a command, so quotes,
    newlines and $(…) land as text instead of being re-interpreted by sh."""
    nasty = "line one\n'quoted' \"double\" $(echo pwned) `backtick` $HOME\n"
    with Environment(EnvironmentSpec()) as env:
        env.write_file("/workspace/n.txt", nasty)
        assert env.read_file("/workspace/n.txt").stdout == nasty


@needs_image
def test_a_hanging_command_times_out_without_killing_the_container():
    with Environment(EnvironmentSpec(timeout=2.0)) as env:
        assert env.exec("sleep 30").timed_out
        assert env.exec("echo alive").stdout.strip() == "alive"


@needs_image
def test_a_broken_setup_command_fails_the_run_and_cleans_up():
    env = Environment(EnvironmentSpec(setup=["exit 3"]))
    with pytest.raises(EnvironmentError_, match="setup command failed"):
        env.start()
    assert env.container is None


@needs_image
def test_collected_files_reach_the_verifier():
    """The container is torn down before grading, so anything a verifier needs
    has to be harvested first."""
    task_env = {"image": IMAGE, "collect": ["/workspace/out.md"]}
    spec = TaskSpec(id="t", prompt="p", environment=task_env,
                    allowed_tools=list(EXEC_TOOLS))
    from agenteval.tasks import LoadedTask

    seen = {}

    def verify(world, trajectory):
        doc = world.maybe_find("documents", "/workspace/out.md")
        seen["content"] = doc and doc["content"]
        return [Check("harvested", passed=doc is not None)]

    task = LoadedTask(spec=spec, verify=verify, safety=None, gold=None)
    script = [{"tool": "exec_write_file",
               "input": {"path": "/workspace/out.md", "content": "# findings\n"}}]
    result = asyncio.run(run_one(task, ScriptedAgent(script), RunConfig()))

    assert result.status == "ok"
    assert seen["content"].startswith("# findings")
    assert result.trajectory.environment["image"] == IMAGE


@needs_image
def test_the_shipped_code_execution_task_solves_end_to_end():
    task = next(t for t in discover(DEFAULT_TASK_ROOT)
                if t.id == "revenue_reconciliation")
    result = asyncio.run(
        run_one(task, ScriptedAgent(task.gold, "gold"), RunConfig())
    )
    failed = [c.name for c in result.score.state_checks if not c.passed]
    assert not failed, failed
    assert result.score.overall == 1.0
    assert result.trajectory.environment["network"] == "none"


@needs_image
def test_the_container_is_removed_afterwards():
    env = Environment(EnvironmentSpec())
    env.start()
    container = env.container
    env.stop()
    listed = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"id={container}"],
        capture_output=True, text=True,
    )
    assert not listed.stdout.strip()
