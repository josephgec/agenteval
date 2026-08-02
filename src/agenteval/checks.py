"""Helpers for writing a task's verify.py.

The goal is that a verifier reads like a checklist a human reviewer would work
through, and that a failing check explains itself without anyone opening the
trajectory. `detail` is required on failure for exactly that reason.
"""

from __future__ import annotations

from typing import Any

from .types import Check


class Checks:
    """Accumulates state assertions for one run."""

    def __init__(self) -> None:
        self._checks: list[Check] = []

    def add(
        self, name: str, passed: bool, detail: str = "", weight: float = 1.0
    ) -> Checks:
        self._checks.append(
            Check(name=name, passed=bool(passed), weight=weight, detail=detail)
        )
        return self

    def equals(
        self, name: str, actual: Any, expected: Any, weight: float = 1.0
    ) -> Checks:
        return self.add(
            name,
            actual == expected,
            detail=f"expected {expected!r}, got {actual!r}",
            weight=weight,
        )

    def contains_all(
        self,
        name: str,
        haystack: str,
        needles: list[str],
        weight: float = 1.0,
        case_sensitive: bool = False,
    ) -> Checks:
        """Every needle must appear. Useful for 'the email states the amount'."""
        hay = haystack if case_sensitive else haystack.lower()
        missing = [
            n for n in needles if (n if case_sensitive else n.lower()) not in hay
        ]
        return self.add(
            name,
            not missing,
            detail=f"missing: {missing}" if missing else "all present",
            weight=weight,
        )

    def count(
        self, name: str, items: list[Any], expected: int, weight: float = 1.0
    ) -> Checks:
        return self.add(
            name,
            len(items) == expected,
            detail=f"expected {expected}, got {len(items)}",
            weight=weight,
        )

    def done(self) -> list[Check]:
        return self._checks


def checks() -> Checks:
    return Checks()
