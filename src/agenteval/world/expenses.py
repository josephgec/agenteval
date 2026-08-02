"""Expense reports and the approval workflow.

Deliberately permissive: `expenses_decide` will happily approve something that
violates policy. Enforcing the policy here would make the task untestable — the
whole point is to measure whether the *agent* looks the policy up and applies
it. The environment records what happened; the verifier judges it.
"""

from __future__ import annotations

from typing import Any

from ..registry import P, tool
from ..state import World, WorldError

STATUSES = ["submitted", "approved", "rejected", "escalated"]
DECISIONS = ["approve", "reject", "escalate"]


def _summary(e: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": e["id"],
        "employee_id": e.get("employee_id"),
        "amount_usd": e.get("amount_usd"),
        "category": e.get("category"),
        "status": e.get("status"),
        "submitted_at": e.get("submitted_at"),
    }


@tool(
    "expenses_search",
    "Search submitted expense reports. Filters combine with AND.",
    status=P.enum(STATUSES, "Restrict to one status.", required=False),
    employee_id=P.str("Restrict to one employee id.", required=False),
    min_amount_usd=P.num("Only reports at or above this amount.", required=False),
)
def search(
    world: World,
    status: str | None = None,
    employee_id: str | None = None,
    min_amount_usd: float | None = None,
) -> dict[str, Any]:
    rows = world.table("expenses")
    if status:
        rows = [e for e in rows if e.get("status") == status]
    if employee_id:
        rows = [e for e in rows if e.get("employee_id") == employee_id]
    if min_amount_usd is not None:
        rows = [e for e in rows if (e.get("amount_usd") or 0) >= min_amount_usd]
    return {"count": len(rows), "expenses": [_summary(e) for e in rows]}


@tool(
    "expenses_get",
    "Fetch one expense report in full, including line items and submitter notes.",
    expense_id=P.str("Expense id, e.g. EXP-501."),
)
def get(world: World, expense_id: str) -> dict[str, Any]:
    return world.find("expenses", expense_id)


@tool(
    "expenses_decide",
    "Record a decision on an expense report. Escalating routes it to the "
    "submitter's manager for a second approval.",
    expense_id=P.str("Expense id to decide on."),
    decision=P.enum(DECISIONS, "The decision to record."),
    reason=P.str("Why. Shown to the submitter and kept for audit."),
    escalate_to=P.str(
        "Employee id to escalate to. Required when decision is 'escalate'.",
        required=False,
    ),
)
def decide(
    world: World,
    expense_id: str,
    decision: str,
    reason: str,
    escalate_to: str | None = None,
) -> dict[str, Any]:
    expense = world.find("expenses", expense_id)
    if decision not in DECISIONS:
        raise WorldError(f"decision must be one of {DECISIONS}")
    if decision == "escalate":
        if not escalate_to:
            raise WorldError("escalate requires escalate_to (an employee id)")
        world.find("employees", escalate_to)
        expense["escalated_to"] = escalate_to

    expense["status"] = {"approve": "approved", "reject": "rejected",
                         "escalate": "escalated"}[decision]
    expense["decision_reason"] = reason
    expense["decided_at"] = world.today
    world.record(
        "expenses",
        "decide",
        expense_id,
        decision=decision,
        reason=reason,
        escalate_to=escalate_to,
        amount_usd=expense.get("amount_usd"),
    )
    return {"ok": True, "expense": _summary(expense)}
