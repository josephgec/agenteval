"""The `Benchmark` protocol: where tasks come from.

`LoadedTask` is what the runner needs — a prompt, a world, a container spec and
something that grades what happened. This module is about *supplying* those.
Reading a directory of hand-written tasks is one way; downloading SWE-bench and
turning 300 GitHub issues into tasks is another. Both end at `LoadedTask`, so
the runner, the grading, the report and the UI never learn which they are
looking at.

That is the whole design. A benchmark adapter is a class with three methods,
and adding one touches no existing code:

    class MyBenchmark:
        name = "mine"
        def prepare(self) -> None: ...            # download, cache, build
        def instance_ids(self) -> list[str]: ...  # what is in it
        def load(self, instance_id) -> LoadedTask

`prepare()` is separate from `load()` because downloading is slow, shared and
worth doing once, while loading happens per instance and must be cheap. It is
required to be idempotent — the second call on a warm cache does nothing.

Instance ids are strings the benchmark chooses. They are used for selection
(`--task`), so they should be the benchmark's own native ids: `HumanEval/12`,
`django__django-11099`. Rewriting them to be prettier makes results
uncomparable with everyone else's.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from ..tasks import LoadedTask


class BenchmarkError(Exception):
    """A benchmark could not be prepared or loaded."""


@runtime_checkable
class Benchmark(Protocol):
    """A source of tasks."""

    #: Stable identifier, used on the command line and recorded in results.
    name: str

    def prepare(self) -> None:
        """Fetch and cache whatever the benchmark needs. Idempotent."""

    def instance_ids(self) -> list[str]:
        """Every instance, in the benchmark's own order."""

    def load(self, instance_id: str) -> LoadedTask:
        """Build one task. Called after `prepare()`."""


def cache_root() -> Path:
    """Where downloaded benchmarks live.

    Outside the repository on purpose: benchmark data is large, is not ours,
    and must not end up committed — this one is a public repo, and a task file
    pushed to GitHub is a task file in the next model's training set.
    """
    return Path.home() / ".cache" / "agenteval"


def select(
    ids: list[str],
    only: list[str] | None = None,
    limit: int | None = None,
    seed: int | None = None,
) -> list[str]:
    """Choose which instances to run.

    Sampling is seeded rather than "the first N". Benchmarks are frequently
    ordered by something — difficulty, repository, date — so a prefix is a
    biased sample that still looks like a score for the whole benchmark. A
    seed keeps it reproducible while `--limit 20` stays honest about being a
    sample.
    """
    if only:
        known = set(ids)
        unknown = [i for i in only if i not in known]
        if unknown:
            # Listing what there is turns a typo into a one-line fix, but a
            # benchmark with 164 instances would bury the message — so the
            # count stands in once a list stops being readable.
            available = (
                f"Available: {sorted(ids)}"
                if len(ids) <= 25
                else f"{len(ids)} instances available; see `agenteval list`"
            )
            raise BenchmarkError(
                f"unknown instance(s) {unknown[:5]}"
                + (f" and {len(unknown) - 5} more" if len(unknown) > 5 else "")
                + f". {available}"
            )
        ids = [i for i in ids if i in set(only)]
    if limit is not None and limit < len(ids):
        ids = sorted(random.Random(seed).sample(ids, limit), key=ids.index)
    return ids


def load_tasks(
    benchmark: Benchmark,
    only: list[str] | None = None,
    limit: int | None = None,
    seed: int | None = None,
) -> list[LoadedTask]:
    """Prepare a benchmark and load the instances asked for."""
    benchmark.prepare()
    return [benchmark.load(i) for i in select(benchmark.instance_ids(), only, limit, seed)]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

#: name -> factory taking the optional `:argument` suffix.
_BENCHMARKS: dict[str, tuple[Callable[[str | None], Benchmark], str]] = {}


def register(
    name: str, factory: Callable[[str | None], Benchmark], description: str = ""
) -> None:
    _BENCHMARKS[name] = (factory, description)


def registered() -> dict[str, str]:
    return {name: description for name, (_, description) in _BENCHMARKS.items()}


def resolve(spec: str) -> Benchmark:
    """Build a benchmark from `name` or `name:argument`.

    The argument means whatever the benchmark wants — a split for SWE-bench, a
    directory for the local one. Deliberately one opaque string: a benchmark
    that needs richer configuration should be constructed in Python rather than
    growing a second configuration language on the command line.
    """
    name, _, argument = spec.partition(":")
    if name not in _BENCHMARKS:
        raise BenchmarkError(
            f"unknown benchmark {name!r}. Available: {sorted(_BENCHMARKS)}"
        )
    factory, _ = _BENCHMARKS[name]
    return factory(argument or None)


def summarise(benchmark: Benchmark) -> dict[str, Any]:
    """What gets recorded alongside the results.

    A score is meaningless without the benchmark and subset it came from, and
    "we ran 20 of SWE-bench Lite" is not the same claim as "we ran SWE-bench
    Lite".
    """
    return {"name": benchmark.name, "instances": len(benchmark.instance_ids())}
