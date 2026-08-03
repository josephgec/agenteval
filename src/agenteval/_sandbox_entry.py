"""Entrypoint inside the sandbox container. Runs untrusted task code.

Everything here executes with no network, no environment, a read-only root and
no capabilities — see `agenteval.sandbox` for how the container is started.
Nothing in this module should acquire privileges or reach outside the process.

Protocol: one JSON request on stdin, one JSON response on stdout.

    {"op": "load",  "task_id": …, "verify_source": …}
    {"op": "grade", "task_id": …, "verify_source": …, "world": …,
     "trajectory": …}

Responses are always `{"ok": bool, …}`. A failure inside task code is reported
rather than raised, so the harness can mark that one run a harness_error and
keep the rest of the suite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from typing import Any


def _load_module(task_id: str, source: str):
    """Execute the task's verify.py in this process.

    Deliberately in-process *here* — the isolation is the container boundary,
    not anything inside it. Running it in yet another subprocess would add no
    security and a great deal of confusion.
    """
    spec = importlib.util.spec_from_loader(f"agenteval_task_{task_id}", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = f"{task_id}/verify.py"
    sys.modules[spec.name] = module
    exec(compile(source, f"{task_id}/verify.py", "exec"), module.__dict__)
    return module


def _handle(request: dict[str, Any]) -> dict[str, Any]:
    module = _load_module(request["task_id"], request["verify_source"])

    if request["op"] == "load":
        if not hasattr(module, "verify"):
            return {"ok": False, "error": "verify.py defines no verify()"}
        return {
            "ok": True,
            "gold": getattr(module, "GOLD", None),
            "has_safety": hasattr(module, "safety"),
        }

    if request["op"] == "grade":
        from .state import World
        from .types import Trajectory

        world = World.from_dict(request["world"])
        trajectory = Trajectory.from_dict(request["trajectory"])

        # Both are produced in one call: the container costs far more than the
        # work inside it, so verify and safety share a single crossing.
        checks = module.verify(world, trajectory)
        violations = (
            list(module.safety(world, trajectory))
            if hasattr(module, "safety")
            else []
        )
        return {
            "ok": True,
            "checks": [
                {"name": c.name, "passed": bool(c.passed),
                 "weight": float(c.weight), "detail": c.detail}
                for c in checks
            ],
            "violations": [str(v) for v in violations],
        }

    return {"ok": False, "error": f"unknown op {request['op']!r}"}


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001
        json.dump({"ok": False, "error": f"unreadable request: {exc}"}, sys.stdout)
        return 0
    try:
        response = _handle(request)
    except Exception:  # noqa: BLE001 - task code failing is a result, not a crash
        response = {"ok": False, "error": traceback.format_exc(limit=6)}
    json.dump(response, sys.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised in the container
    raise SystemExit(main())
