"""Helpers for writing a task's verify.py.

The goal is that a verifier reads like a checklist a human reviewer would work
through, and that a failing check explains itself without anyone opening the
trajectory. `detail` is required on failure for exactly that reason.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from .types import Check


def _ordinal(day: int) -> str:
    if 11 <= day <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def date_renderings(iso_date: str) -> list[str]:
    """Every reasonable way a person might write a date in prose.

    Asserting on the ISO literal alone marks down an agent for writing
    "12 September 2026" to a customer, which is better prose than
    "2026-09-12" — the check would be demanding a worse answer.
    """
    parsed = date.fromisoformat(iso_date)
    day, year = parsed.day, parsed.year
    suffix = _ordinal(day)
    variants = {iso_date, parsed.strftime("%d/%m/%Y"), parsed.strftime("%Y/%m/%d")}
    for month in (parsed.strftime("%B"), parsed.strftime("%b")):
        variants |= {
            f"{day} {month} {year}",
            f"{day:02d} {month} {year}",
            f"{day}{suffix} {month} {year}",
            f"{month} {day}, {year}",
            f"{month} {day} {year}",
            f"{month} {day}{suffix}, {year}",
        }
    return sorted(variants)


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

    def states_date(
        self, name: str, haystack: str, iso_date: str, weight: float = 1.0
    ) -> Checks:
        """Assert the text gives this date, in any conventional format."""
        accepted = date_renderings(iso_date)
        hay = haystack.lower()
        found = next((v for v in accepted if v.lower() in hay), None)
        return self.add(
            name,
            found is not None,
            detail=(
                f"found {found!r}"
                if found
                else f"no rendering of {iso_date} present (accepts e.g. "
                f"{', '.join(accepted[:3])})"
            ),
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
