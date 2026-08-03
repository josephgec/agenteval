"""The hand-written task directory, as a benchmark.

This exists to keep the protocol honest. An abstraction invented for a
downloaded benchmark and never applied to the suite that already works is an
abstraction nobody has tested; if the existing five tasks could not be
expressed through `Benchmark`, the shape would be wrong and this file is where
that would have shown up.

It is a thin wrapper over `discover`, which is the point — nothing about the
local format had to change to fit.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..tasks import DEFAULT_TASK_ROOT, LoadedTask, load_task
from .base import Benchmark, BenchmarkError, register


class LocalBenchmark:
    """Tasks read from a directory of `task.yaml` / `seed.json` / `verify.py`."""

    name = "local"

    def __init__(self, root: str | Path | None = None, sandbox: object = None) -> None:
        self.root = Path(root or DEFAULT_TASK_ROOT)
        #: Verifier isolation, for suites you did not write. Only meaningful
        #: here: a downloaded benchmark's adapter is code in this repository,
        #: not a `verify.py` shipped inside the download.
        self.sandbox = sandbox

    def prepare(self) -> None:
        if not self.root.exists():
            raise BenchmarkError(f"task directory {self.root} does not exist")

    def _index(self) -> dict[str, Path]:
        """Instance id to directory.

        Keyed on the id in `task.yaml` rather than the directory name, because
        those are allowed to differ and `--task` selects on the id the results
        are filed under. Reading one small yaml per task to find out is cheaper
        than a selection that silently misses.
        """
        index: dict[str, Path] = {}
        for directory in sorted(self.root.iterdir()):
            config_path = directory / "task.yaml"
            if not directory.is_dir() or not config_path.exists():
                continue
            config = yaml.safe_load(config_path.read_text()) or {}
            index[config.get("id") or directory.name] = directory
        return index

    def instance_ids(self) -> list[str]:
        return list(self._index())

    def load(self, instance_id: str) -> LoadedTask:
        directory = self._index().get(instance_id, self.root / instance_id)
        task = load_task(directory, sandbox=self.sandbox)
        task.benchmark = self.name
        return task


register(
    "local",
    lambda argument: LocalBenchmark(argument),
    "hand-written enterprise workflow tasks from ./tasks",
)

__all__ = ["LocalBenchmark", "Benchmark"]
