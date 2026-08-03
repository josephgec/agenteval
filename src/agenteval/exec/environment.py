"""A container that tool calls execute inside.

The execution stack, in one paragraph: the agent and `ToolSession` stay on the
host, and a long-lived container per run provides somewhere for tool calls to
actually happen. Three consequences worth stating, because each one is a
decision that would be expensive to reverse later.

*The audit trail stays trustworthy.* `ToolSession` observes from outside the
container, so "the agent reached for a forbidden tool" remains a claim made by
harness code rather than one the sandboxed side reports about itself. Move the
session inside and the safety signal becomes self-attested.

*The credential never enters the container.* The model API is called from the
host, so nothing in the sandbox can read a key. The gateway in `proxy.py`
exists for the opposite direction — benchmarks that need to fetch things —
rather than to keep a key away from the agent.

*One container per run, not per call.* `docker run` costs a few hundred
milliseconds; `docker exec` costs a few. A benchmark instance is a session with
state — a repo, a virtualenv, files written three steps ago — so per-call
containers would be both slower and wrong.

The image is a parameter. Today it is a small Python environment; for a
downloaded benchmark it becomes that benchmark's own image, which is the seam
that keeps this from needing a rewrite.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any

from .proxy import Proxy

DEFAULT_IMAGE = "agenteval-exec:latest"


class EnvironmentError_(Exception):
    """The environment itself failed — not the command that ran in it."""


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def render(self, limit: int = 4000) -> str:
        """What the agent sees. Shaped like a terminal, because that is the
        thing it is being asked to reason about."""
        if self.timed_out:
            return f"Timed out.\n{self.stdout[-limit:]}"
        parts = []
        if self.stdout:
            parts.append(self.stdout[-limit:])
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr[-limit:]}")
        if self.exit_code != 0:
            parts.append(f"[exit {self.exit_code}]")
        return "\n".join(parts) or "(no output)"


@dataclass
class EnvironmentSpec:
    """How a task's execution container is built.

    Declared per task, so a benchmark that needs its own image, more memory or
    (deliberately) network access says so rather than the harness guessing.
    """

    image: str = DEFAULT_IMAGE
    #: "none" by default. A benchmark that genuinely needs egress opts in, and
    #: that choice then shows up in the task definition where it is reviewable.
    network: str = "none"
    #: Hosts the workspace may reach, through a logging gateway. Naming any of
    #: them replaces `network` with an internal network whose only exit is that
    #: gateway — see proxy.py. Empty keeps whatever `network` says.
    allow_hosts: list[str] = field(default_factory=list)
    memory: str = "1g"
    cpus: str = "2.0"
    workdir: str = "/workspace"
    #: Per-command ceiling. A build that hangs should fail its own run.
    timeout: float = 120.0
    #: Wall-clock ceiling for the whole container, as a backstop.
    lifetime: float = 1800.0
    #: Files written into the container before setup runs — input data, a
    #: fixture, a repo patch. Path to contents.
    files: dict[str, str] = field(default_factory=dict)
    #: Commands run once after start — clone a repo, install dependencies.
    setup: list[str] = field(default_factory=list)
    #: Paths copied back out before teardown. Without this a verifier cannot
    #: see anything the agent wrote, because the container is gone by the time
    #: grading runs. Collected files land in the world as documents, so the
    #: existing artifact selectors, verifiers and report all reach them
    #: unchanged.
    collect: list[str] = field(default_factory=list)
    #: Workspace size. Repositories and build artifacts need more than data.
    workspace_mb: int = 512
    #: None uses the image's own user. Benchmark images frequently assume root.
    user: str | None = None
    read_only_root: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> EnvironmentSpec | None:
        if not config:
            return None
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(config) - known
        if unknown:
            raise EnvironmentError_(
                f"unknown environment settings {sorted(unknown)}; "
                f"accepted: {sorted(known)}"
            )
        spec = cls(**config)
        if spec.allow_hosts and config.get("network") not in (None, "none"):
            # An allowlist beside a network that already grants unfiltered
            # egress is not a stricter policy, it is a misleading one. Anyone
            # reading the task would take the list for the boundary.
            raise EnvironmentError_(
                f"allow_hosts cannot be combined with network: {spec.network!r}, "
                "which already grants unrestricted egress. Remove `network` and "
                "the allowlist becomes the boundary."
            )
        return spec


class Environment:
    """A running container. Use as a context manager."""

    def __init__(self, spec: EnvironmentSpec, docker: str = "docker") -> None:
        self.spec = spec
        self.docker = docker
        self.container: str | None = None
        #: The egress gateway, for tasks that named hosts. None otherwise.
        self.proxy: Proxy | None = None
        #: Kept across teardown so the record survives the gateway it came from.
        self._egress: dict[str, Any] | None = None
        #: Every command run, for the record. The trajectory logs tool calls;
        #: this logs what actually executed, including setup the agent never saw.
        self.log: list[dict[str, Any]] = []

    # -- lifecycle ---------------------------------------------------------- #

    def _run_args(self) -> list[str]:
        name = f"agenteval-{uuid.uuid4().hex[:12]}"
        args = [
            self.docker, "run", "--detach", "--name", name,
            # With a gateway this is the internal network, which has no route
            # off the host. The allowlist is enforced by there being nowhere
            # else to go, not by the variables set below.
            "--network", self.proxy.network if self.proxy else self.spec.network,
            # No host environment crosses in, so a credential on this machine
            # is not reachable from code the agent generates.
            "--env-file", "/dev/null",
            "--memory", self.spec.memory,
            "--cpus", str(self.spec.cpus),
            "--pids-limit", "512",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--workdir", self.spec.workdir,
        ]
        if self.spec.read_only_root:
            # Writable where work happens, read-only everywhere else.
            # mode=1777 (sticky, world-writable — the /tmp convention) so the
            # workspace is usable whatever uid the image runs as. A tmpfs
            # otherwise arrives owned by root, which a non-root image cannot
            # write to, and pinning a uid here would break every benchmark
            # image that picked a different one.
            args += [
                "--read-only",
                "--tmpfs",
                f"{self.spec.workdir}:rw,exec,mode=1777,size={self.spec.workspace_mb}m",
                "--tmpfs", "/tmp:rw,exec,mode=1777,size=256m",
            ]
        if self.proxy:
            for variable in self.proxy.environment_variables():
                args += ["--env", variable]
        if self.spec.user:
            args += ["--user", self.spec.user]
        # Held open so exec has something to attach to; the container does no
        # work of its own.
        args += [self.spec.image, "sleep", str(int(self.spec.lifetime))]
        return args

    def start(self) -> None:
        if self.container:
            return
        if self.spec.allow_hosts:
            # Before the workspace, so the network exists to attach it to and
            # the gateway is already answering when the first request lands.
            self.proxy = Proxy(
                self.spec.allow_hosts, image=self.spec.image,
                docker=self.docker, lifetime=self.spec.lifetime,
            )
            self.proxy.start()
        done = subprocess.run(self._run_args(), capture_output=True, text=True)
        if done.returncode != 0:
            raise EnvironmentError_(
                f"could not start {self.spec.image!r}: {done.stderr.strip()[:600]}"
            )
        self.container = done.stdout.strip()
        for path, content in self.spec.files.items():
            result = self.write_file(path, content)
            if result.exit_code != 0:
                self.stop()
                raise EnvironmentError_(
                    f"could not seed {path}: {result.stderr[-400:]}"
                )
        for command in self.spec.setup:
            result = self.exec(command)
            self.log.append({"phase": "setup", "command": command,
                             "exit_code": result.exit_code})
            if not result.ok:
                self.stop()
                raise EnvironmentError_(
                    f"setup command failed ({result.exit_code}): {command}\n"
                    f"{result.stderr[-600:]}"
                )

    def stop(self) -> None:
        # The gateway goes second: the network cannot be removed while the
        # workspace is still attached to it, and `snapshot()` may have been
        # called just before this.
        if self.container:
            subprocess.run(
                [self.docker, "rm", "--force", self.container],
                capture_output=True, text=True,
            )
            self.container = None
        if self.proxy:
            self._egress = self.proxy.snapshot()
            self.proxy.stop()
            self.proxy = None

    def __enter__(self) -> Environment:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- work --------------------------------------------------------------- #

    def exec(self, command: str, timeout: float | None = None) -> ExecResult:
        """Run a shell command inside the container."""
        if not self.container:
            raise EnvironmentError_("environment is not running")
        limit = timeout or self.spec.timeout
        try:
            done = subprocess.run(
                [self.docker, "exec", self.container, "sh", "-c", command],
                capture_output=True, text=True, timeout=limit,
            )
        except subprocess.TimeoutExpired as exc:
            # A hung command must not take the suite with it. The container is
            # still usable, so the agent can try something else.
            return ExecResult(
                exit_code=124,
                stdout=(exc.stdout or b"").decode(errors="replace")
                if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=f"command exceeded {limit:g}s",
                timed_out=True,
            )
        result = ExecResult(done.returncode, done.stdout, done.stderr)
        self.log.append({"phase": "exec", "command": command,
                         "exit_code": result.exit_code})
        return result

    def write_file(self, path: str, content: str) -> ExecResult:
        """Write a file without going through the shell.

        Piped on stdin rather than interpolated into a command, so content
        containing quotes, newlines or `$(…)` lands verbatim instead of being
        re-interpreted by sh.
        """
        if not self.container:
            raise EnvironmentError_("environment is not running")
        done = subprocess.run(
            [self.docker, "exec", "-i", self.container, "sh", "-c",
             f"mkdir -p \"$(dirname {shlex.quote(path)})\" && "
             f"cat > {shlex.quote(path)}"],
            input=content, capture_output=True, text=True,
            timeout=self.spec.timeout,
        )
        self.log.append({"phase": "write", "command": path,
                         "exit_code": done.returncode})
        return ExecResult(done.returncode, "", done.stderr)

    def read_file(self, path: str, limit: int = 200_000) -> ExecResult:
        return self.exec(f"head -c {limit} {shlex.quote(path)}")

    def harvest(self) -> dict[str, str]:
        """Read the paths this task asked to keep, before the container dies.

        Missing files are simply absent rather than an error — an agent that
        never wrote its report should fail a check about the report, not
        produce a harness error that obscures it.
        """
        collected: dict[str, str] = {}
        for path in self.spec.collect:
            result = self.read_file(path)
            if result.ok and result.stdout:
                collected[path] = result.stdout
        return collected

    def snapshot(self) -> dict[str, Any]:
        """What ran, for the result record."""
        egress = self.proxy.snapshot() if self.proxy else self._egress
        return {
            "image": self.spec.image,
            # What the workspace actually had, which for a proxied task is not
            # what `network:` said — reporting "none" there would be a lie in
            # the one direction that matters.
            "network": "proxy" if self.spec.allow_hosts else self.spec.network,
            "commands": len([e for e in self.log if e["phase"] == "exec"]),
            "log": self.log,
            "egress": egress,
        }


def available(docker: str = "docker") -> bool:
    try:
        return subprocess.run(
            [docker, "info"], capture_output=True, timeout=20
        ).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def image_present(image: str, docker: str = "docker") -> bool:
    try:
        return subprocess.run(
            [docker, "image", "inspect", image], capture_output=True, timeout=30
        ).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def describe(spec: EnvironmentSpec) -> str:
    return json.dumps(
        {
            "image": spec.image,
            "network": "proxy" if spec.allow_hosts else spec.network,
            "memory": spec.memory,
        },
        sort_keys=True,
    )
