"""Support ticketing."""

from __future__ import annotations

from typing import Any

from ..registry import P, tool
from ..state import World, WorldError

STATUSES = ["open", "in_progress", "waiting_customer", "resolved", "closed"]
PRIORITIES = ["P0", "P1", "P2", "P3"]
TEAMS = ["support", "engineering", "it", "billing", "security", "sales"]


def _summary(t: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": t["id"],
        "subject": t["subject"],
        "status": t.get("status"),
        "priority": t.get("priority"),
        "team": t.get("team"),
        "assignee": t.get("assignee"),
        "account_id": t.get("account_id"),
        "created_at": t.get("created_at"),
    }


@tool(
    "tickets_search",
    "Search support tickets. Filters are optional and combine with AND. "
    "Returns summaries; use tickets.get for the body and comments.",
    status=P.enum(STATUSES, "Restrict to one status.", required=False),
    priority=P.enum(PRIORITIES, "Restrict to one priority.", required=False),
    team=P.enum(TEAMS, "Restrict to one owning team.", required=False),
    account_id=P.str("Restrict to one account id.", required=False),
    unassigned_only=P.bool(
        "If true, only tickets with no assignee.", required=False
    ),
)
def search(
    world: World,
    status: str | None = None,
    priority: str | None = None,
    team: str | None = None,
    account_id: str | None = None,
    unassigned_only: bool | None = None,
) -> dict[str, Any]:
    rows = world.table("tickets")
    if status:
        rows = [t for t in rows if t.get("status") == status]
    if priority:
        rows = [t for t in rows if t.get("priority") == priority]
    if team:
        rows = [t for t in rows if t.get("team") == team]
    if account_id:
        rows = [t for t in rows if t.get("account_id") == account_id]
    if unassigned_only:
        rows = [t for t in rows if not t.get("assignee")]
    return {"count": len(rows), "tickets": [_summary(t) for t in rows]}


@tool(
    "tickets_get",
    "Fetch one ticket in full, including body and comment thread.",
    ticket_id=P.str("Ticket id, e.g. TKT-1042."),
)
def get(world: World, ticket_id: str) -> dict[str, Any]:
    return world.find("tickets", ticket_id)


@tool(
    "tickets_create",
    "File a new support ticket.",
    subject=P.str("Short one-line summary."),
    body=P.str("Full description of the issue or request."),
    team=P.enum(TEAMS, "Team that should own this ticket."),
    priority=P.enum(PRIORITIES, "Priority: P0 is highest, P3 lowest."),
    account_id=P.str("Related customer account id, if any.", required=False),
    assignee=P.str("Employee id to assign to, if known.", required=False),
)
def create(
    world: World,
    subject: str,
    body: str,
    team: str,
    priority: str,
    account_id: str | None = None,
    assignee: str | None = None,
) -> dict[str, Any]:
    if account_id:
        world.find("accounts", account_id)
    if assignee:
        world.find("employees", assignee)
    ticket = {
        "id": world.next_id("TKT-NEW"),
        "subject": subject,
        "body": body,
        "team": team,
        "priority": priority,
        "status": "open",
        "assignee": assignee,
        "account_id": account_id,
        "created_at": world.today,
        "comments": [],
    }
    world.insert("tickets", ticket)
    world.record(
        "tickets",
        "create",
        ticket["id"],
        subject=subject,
        team=team,
        priority=priority,
        account_id=account_id,
    )
    return {"ok": True, "ticket": _summary(ticket)}


@tool(
    "tickets_update",
    "Update an existing ticket. Only the fields you pass are changed.",
    ticket_id=P.str("Ticket id to update."),
    status=P.enum(STATUSES, "New status.", required=False),
    priority=P.enum(PRIORITIES, "New priority.", required=False),
    team=P.enum(TEAMS, "Reassign to this team.", required=False),
    assignee=P.str("Employee id to assign to.", required=False),
)
def update(
    world: World,
    ticket_id: str,
    status: str | None = None,
    priority: str | None = None,
    team: str | None = None,
    assignee: str | None = None,
) -> dict[str, Any]:
    ticket = world.find("tickets", ticket_id)
    changed: dict[str, Any] = {}
    for field_name, value, allowed in (
        ("status", status, STATUSES),
        ("priority", priority, PRIORITIES),
        ("team", team, TEAMS),
    ):
        if value is not None:
            if value not in allowed:
                raise WorldError(f"{field_name} must be one of {allowed}")
            ticket[field_name] = value
            changed[field_name] = value
    if assignee is not None:
        world.find("employees", assignee)
        ticket["assignee"] = assignee
        changed["assignee"] = assignee
    if not changed:
        raise WorldError("tickets.update requires at least one field to change")
    world.record("tickets", "update", ticket_id, **changed)
    return {"ok": True, "ticket": _summary(ticket)}


@tool(
    "tickets_comment",
    "Append a comment to a ticket's thread. Visible to the customer.",
    ticket_id=P.str("Ticket id to comment on."),
    body=P.str("Comment text."),
)
def comment(world: World, ticket_id: str, body: str) -> dict[str, Any]:
    ticket = world.find("tickets", ticket_id)
    entry = {"author": "agent", "body": body, "at": world.today}
    ticket.setdefault("comments", []).append(entry)
    world.record("tickets", "comment", ticket_id, body=body)
    return {"ok": True, "ticket_id": ticket_id, "comment_count": len(ticket["comments"])}
