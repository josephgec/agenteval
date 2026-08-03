"""Running task code in a container.

`verify.py` is arbitrary Python. Loading a task executes it, so a task suite you
did not write is untrusted code running in your interpreter — and the harness
otherwise trusts tasks completely while trusting the agent not at all.

What is containerised, and what is not
-------------------------------------
Only the task code. The agent loop stays on the host, and so does the API key.
Putting the whole run in one container would place a malicious verifier next to
your credential, which is most of what you were trying to prevent. The sandbox
therefore gets **no network and no environment**: nothing to steal, and nowhere
to send it.

The world crosses as JSON and comes back as checks and violations. That is the
entire interface — a verifier cannot reach the agent, the API, the filesystem,
or the other runs in the suite.

Cost: one container per graded run, a few hundred milliseconds. Verify and
safety share a single crossing because the container dominates the cost.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import Check, Trajectory
from .state import World

IMAGE = "agenteval-sandbox:latest"
DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile"


class SandboxError(Exception):
    """Raised when the sandbox itself fails — not when task code does."""


#: Container flags. Each of these is load-bearing; this list is the security
#: boundary, so it is kept in one place rather than spread through call sites.
HARDENING = [
    # Nothing to exfiltrate to. The single most important flag here.
    "--network", "none",
    # No host env crosses the boundary, so ANTHROPIC_API_KEY is unreachable
    # even though the same machine is holding one.
    "--env-file", "/dev/null",
    "--read-only",
    "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
    "--user", "65534:65534",          # nobody
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    "--pids-limit", "256",            # fork bombs
    "--memory", "512m",
    "--cpus", "1.0",
]


@dataclass
class Sandbox:
    """Runs a task's verify.py inside a container."""

    image: str = IMAGE
    #: Wall-clock ceiling per call. A verifier that spins forever should fail
    #: its own run, not wedge the suite.
    timeout: float = 60.0
    docker: str = "docker"
    #: One-entry cache. The runner calls verify() then safety() with the same
    #: world and trajectory, and both come back from a single crossing.
    _last: tuple[Any, Any, dict[str, Any]] | None = field(
        default=None, repr=False, compare=False
    )

    # -- availability ------------------------------------------------------- #

    def preflight(self) -> None:
        """Fail early and legibly rather than mid-suite."""
        if shutil.which(self.docker) is None:
            raise SandboxError(
                f"{self.docker!r} is not on PATH. Install Docker, or drop "
                "--sandbox to run task code in-process."
            )
        probe = subprocess.run(
            [self.docker, "image", "inspect", self.image],
            capture_output=True, text=True,
        )
        if probe.returncode != 0:
            raise SandboxError(
                f"image {self.image!r} is not built. Run "
                "`agenteval sandbox build` first."
            )

    def build(self) -> None:
        if not DOCKERFILE.exists():
            raise SandboxError(f"no Dockerfile at {DOCKERFILE}")
        done = subprocess.run(
            [self.docker, "build", "-t", self.image, "-f", str(DOCKERFILE),
             str(DOCKERFILE.parent)],
            capture_output=True, text=True,
        )
        if done.returncode != 0:
            raise SandboxError(f"docker build failed:\n{done.stderr[-2000:]}")

    # -- protocol ----------------------------------------------------------- #

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            done = subprocess.run(
                [self.docker, "run", "--rm", "-i", *HARDENING, self.image],
                input=json.dumps(request),
                capture_output=True, text=True, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise SandboxError(
                f"task code exceeded {self.timeout:g}s in the sandbox"
            ) from None
        if done.returncode != 0:
            raise SandboxError(
                f"sandbox exited {done.returncode}: {done.stderr[-800:]}"
            )
        try:
            response = json.loads(done.stdout)
        except json.JSONDecodeError:
            raise SandboxError(
                f"sandbox returned no JSON: {done.stdout[:400]!r}"
            ) from None
        if not response.get("ok"):
            raise SandboxError(response.get("error", "task code failed"))
        return response

    # -- operations --------------------------------------------------------- #

    def load(self, task_id: str, source: str) -> dict[str, Any]:
        """Read GOLD and capability flags without executing anything locally."""
        return self._call(
            {"op": "load", "task_id": task_id, "verify_source": source}
        )

    def grade(
        self, task_id: str, source: str, world: World, trajectory: Trajectory
    ) -> dict[str, Any]:
        if self._last is not None:
            cached_world, cached_traj, response = self._last
            if cached_world is world and cached_traj is trajectory:
                return response
        response = self._call({
            "op": "grade",
            "task_id": task_id,
            "verify_source": source,
            "world": world.to_dict(),
            "trajectory": trajectory.to_dict(),
        })
        self._last = (world, trajectory, response)
        return response


class SandboxedGrader:
    """Adapts a sandboxed task to the plain `verify` / `safety` interface.

    The runner is unaware that grading crossed a container boundary, which is
    what keeps sandboxing a deployment choice rather than a second code path.
    """

    def __init__(self, sandbox: Sandbox, task_id: str, source: str) -> None:
        self.sandbox = sandbox
        self.task_id = task_id
        self.source = source

    def verify(self, world: World, trajectory: Trajectory) -> list[Check]:
        response = self.sandbox.grade(self.task_id, self.source, world, trajectory)
        return [Check(**c) for c in response["checks"]]

    def safety(self, world: World, trajectory: Trajectory) -> list[str]:
        response = self.sandbox.grade(self.task_id, self.source, world, trajectory)
        return list(response["violations"])
