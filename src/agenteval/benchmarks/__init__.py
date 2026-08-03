"""Benchmarks: where tasks come from.

Importing this package registers the built-in adapters, so `resolve("local")`
and `resolve("humaneval")` work without the caller knowing which module defines
them.
"""

from .base import (
    Benchmark,
    BenchmarkError,
    cache_root,
    load_tasks,
    register,
    registered,
    resolve,
    select,
    summarise,
)
from .humaneval import HumanEvalBenchmark
from .local import LocalBenchmark

__all__ = [
    "Benchmark",
    "BenchmarkError",
    "HumanEvalBenchmark",
    "LocalBenchmark",
    "cache_root",
    "load_tasks",
    "register",
    "registered",
    "resolve",
    "select",
    "summarise",
]
