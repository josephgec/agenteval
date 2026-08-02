"""HR directory: employees, reporting lines, and approval authority."""

from __future__ import annotations

from typing import Any

from ..registry import P, tool
from ..state import World


def _summary(e: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": e["id"],
        "name": e["name"],
        "email": e.get("email"),
        "title": e.get("title"),
        "department": e.get("department"),
        "manager_id": e.get("manager_id"),
    }


@tool(
    "hr_search_employees",
    "Search the employee directory. Filters combine with AND.",
    query=P.str(
        "Case-insensitive substring match on name, email, or title.",
        required=False,
    ),
    department=P.str("Exact department name, e.g. Engineering.", required=False),
    manager_id=P.str("Only direct reports of this employee id.", required=False),
)
def search_employees(
    world: World,
    query: str | None = None,
    department: str | None = None,
    manager_id: str | None = None,
) -> dict[str, Any]:
    rows = world.table("employees")
    if query:
        q = query.lower()
        rows = [
            e
            for e in rows
            if q in e.get("name", "").lower()
            or q in e.get("email", "").lower()
            or q in e.get("title", "").lower()
        ]
    if department:
        rows = [e for e in rows if e.get("department") == department]
    if manager_id:
        rows = [e for e in rows if e.get("manager_id") == manager_id]
    return {"count": len(rows), "employees": [_summary(e) for e in rows]}


@tool(
    "hr_get_employee",
    "Fetch one employee's full record, including approval limit and manager "
    "details.",
    employee_id=P.str("Employee id, e.g. EMP-014."),
)
def get_employee(world: World, employee_id: str) -> dict[str, Any]:
    employee = dict(world.find("employees", employee_id))
    if manager_id := employee.get("manager_id"):
        if manager := world.maybe_find("employees", manager_id):
            employee["manager"] = _summary(manager)
    return employee
