"""Loading tasks from disk.

A task is a directory:

    tasks/<id>/
        task.yaml    prompt, limits, rubric, forbidden tools
        seed.json    the world the agent wakes up in
        verify.py    verify(world, trajectory) -> list[Check]
                     safety(world, trajectory) -> list[str]   (optional)
                     GOLD: list[Step]                          (optional)

Keeping the verifier as real Python rather than a declarative assertion format
is deliberate: enterprise workflow success is usually a relationship between
records ("the escalation went to *that employee's* manager"), which a config
language expresses badly.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import yaml

from .types import Check, RubricCriterion, TaskSpec, Trajectory
from .state import World

DEFAULT_TASK_ROOT = Path(__file__).resolve().parents[2] / "tasks"


class TaskError(Exception):
    pass


@dataclass
class LoadedTask:
    spec: TaskSpec
    verify: Callable[[World, Trajectory], list[Check]]
    safety: Callable[[World, Trajectory], list[str]] | None
    gold: list[dict[str, Any]] | None

    @property
    def id(self) -> str:
        return self.spec.id


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise TaskError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered so dataclasses and any relative imports inside resolve.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_task(directory: Path) -> LoadedTask:
    directory = Path(directory)
    config_path = directory / "task.yaml"
    if not config_path.exists():
        raise TaskError(f"{directory} has no task.yaml")

    config = yaml.safe_load(config_path.read_text()) or {}
    task_id = config.get("id") or directory.name

    seed_path = directory / "seed.json"
    seed = json.loads(seed_path.read_text()) if seed_path.exists() else {}

    if "prompt" not in config:
        raise TaskError(f"task {task_id} has no prompt")

    rubric = [
        RubricCriterion(
            id=item["id"],
            description=item["description"],
            weight=float(item.get("weight", 1.0)),
        )
        for item in config.get("rubric", [])
    ]
    if rubric and not config.get("rubric_artifacts"):
        raise TaskError(
            f"task {task_id} defines a rubric but no rubric_artifacts, so the "
            "judge would grade nothing"
        )

    spec = TaskSpec(
        id=task_id,
        prompt=config["prompt"].strip(),
        system=(config.get("system") or None),
        seed=seed,
        allowed_tools=config.get("allowed_tools", []) or [],
        forbidden_tools=config.get("forbidden_tools", []) or [],
        rubric=rubric,
        rubric_artifacts=config.get("rubric_artifacts", []) or [],
        max_steps=int(config.get("max_steps", 40)),
        tags=config.get("tags", []) or [],
        source_dir=str(directory),
    )

    verify_path = directory / "verify.py"
    if not verify_path.exists():
        raise TaskError(f"task {task_id} has no verify.py")
    module = _load_module(verify_path, f"agenteval_task_{task_id}")
    if not hasattr(module, "verify"):
        raise TaskError(f"task {task_id}: verify.py defines no verify()")

    return LoadedTask(
        spec=spec,
        verify=module.verify,
        safety=getattr(module, "safety", None),
        gold=getattr(module, "GOLD", None),
    )


def discover(
    root: Path | str = DEFAULT_TASK_ROOT, only: list[str] | None = None
) -> list[LoadedTask]:
    """Load every task under `root`, optionally filtered to specific ids."""
    root = Path(root)
    if not root.exists():
        raise TaskError(f"task directory {root} does not exist")
    tasks = [
        load_task(d)
        for d in sorted(root.iterdir())
        if d.is_dir() and (d / "task.yaml").exists()
    ]
    if only:
        by_id = {t.id: t for t in tasks}
        unknown = [t for t in only if t not in by_id]
        if unknown:
            raise TaskError(
                f"unknown task(s) {unknown}. Available: {sorted(by_id)}"
            )
        tasks = [by_id[t] for t in only]
    return tasks


def filter_by_tag(tasks: list[LoadedTask], tag: str) -> list[LoadedTask]:
    return [t for t in tasks if tag in t.spec.tags]
