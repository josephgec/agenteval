"""The mutable state of the simulated enterprise.

One World per run, built fresh from a task's seed. Nothing is shared between
runs, so tasks are independently parallelizable and re-runnable.

Every mutation goes through `World.record()`. Verifiers can assert on final
state *or* on the mutation log — "did they set priority to P1" vs "did they set
priority twice, thrashing". Both matter for agentic evaluation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


class WorldError(Exception):
    """Raised by a service when the agent asks for something invalid.

    Surfaced to the agent as a tool_result with is_error=True, not as a crash —
    recovering from a bad call is part of what we are evaluating.
    """


@dataclass
class Mutation:
    service: str
    action: str
    target: str
    payload: dict[str, Any] = field(default_factory=dict)


class World:
    """Holds every service's records plus the mutation log."""

    #: Collections a seed may populate. Anything else in the seed is an error,
    #: which catches typos in task authoring early.
    COLLECTIONS = (
        "accounts",
        "contacts",
        "tickets",
        "inbox",
        "outbox",
        "employees",
        "documents",
        "expenses",
    )

    def __init__(self, seed: dict[str, Any] | None = None) -> None:
        seed = copy.deepcopy(seed or {})
        unknown = set(seed) - {*self.COLLECTIONS, "today"}
        if unknown:
            raise WorldError(f"seed has unknown collections: {sorted(unknown)}")

        self.today: str = seed.get("today", "2026-08-01")
        self.data: dict[str, list[dict[str, Any]]] = {
            name: seed.get(name, []) for name in self.COLLECTIONS
        }
        self.mutations: list[Mutation] = []
        self._counters: dict[str, int] = {}

    # -- collection access -------------------------------------------------- #

    def table(self, name: str) -> list[dict[str, Any]]:
        return self.data[name]

    def find(self, collection: str, record_id: str) -> dict[str, Any]:
        for row in self.data[collection]:
            if row.get("id") == record_id:
                return row
        raise WorldError(
            f"no {collection[:-1]} with id {record_id!r}; "
            f"known ids: {[r.get('id') for r in self.data[collection]][:10]}"
        )

    def maybe_find(self, collection: str, record_id: str) -> dict[str, Any] | None:
        for row in self.data[collection]:
            if row.get("id") == record_id:
                return row
        return None

    def insert(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        self.data[collection].append(record)
        return record

    def next_id(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}-{self._counters[prefix]:04d}"

    # -- audit -------------------------------------------------------------- #

    def record(
        self, service: str, action: str, target: str, **payload: Any
    ) -> None:
        self.mutations.append(Mutation(service, action, target, payload))

    def mutations_for(
        self, service: str | None = None, action: str | None = None
    ) -> list[Mutation]:
        return [
            m
            for m in self.mutations
            if (service is None or m.service == service)
            and (action is None or m.action == action)
        ]

    # -- convenience for verifiers ------------------------------------------ #

    @property
    def outbox(self) -> list[dict[str, Any]]:
        """Emails the agent sent. The most common assertion target."""
        return self.data["outbox"]

    def emails_to(self, address: str) -> list[dict[str, Any]]:
        addr = address.lower()
        return [
            e
            for e in self.outbox
            if addr in [r.lower() for r in e.get("to", [])]
            or addr in [r.lower() for r in e.get("cc", [])]
        ]
