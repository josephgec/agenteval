"""CRM: accounts and contacts."""

from __future__ import annotations

from typing import Any

from ..registry import P, tool
from ..state import World, WorldError

TIERS = ["enterprise", "mid_market", "smb"]
HEALTH = ["green", "yellow", "red"]


def _summary(a: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": a["id"],
        "name": a["name"],
        "tier": a.get("tier"),
        "arr_usd": a.get("arr_usd"),
        "renewal_date": a.get("renewal_date"),
        "health": a.get("health"),
    }


@tool(
    "crm_search_accounts",
    "Search customer accounts. All filters are optional and combine with AND. "
    "Returns summaries; use crm.get_account for the full record.",
    query=P.str("Case-insensitive substring match on account name.", required=False),
    tier=P.enum(TIERS, "Restrict to one account tier.", required=False),
    min_arr_usd=P.num("Only accounts with at least this ARR.", required=False),
    renewal_before=P.str(
        "Only accounts whose renewal_date is strictly before this ISO date "
        "(YYYY-MM-DD).",
        required=False,
    ),
)
def search_accounts(
    world: World,
    query: str | None = None,
    tier: str | None = None,
    min_arr_usd: float | None = None,
    renewal_before: str | None = None,
) -> dict[str, Any]:
    rows = world.table("accounts")
    if query:
        q = query.lower()
        rows = [a for a in rows if q in a["name"].lower()]
    if tier:
        rows = [a for a in rows if a.get("tier") == tier]
    if min_arr_usd is not None:
        rows = [a for a in rows if (a.get("arr_usd") or 0) >= min_arr_usd]
    if renewal_before:
        rows = [
            a
            for a in rows
            if a.get("renewal_date") and a["renewal_date"] < renewal_before
        ]
    return {"count": len(rows), "accounts": [_summary(a) for a in rows]}


@tool(
    "crm_get_account",
    "Fetch one account's full record, including its contacts.",
    account_id=P.str("Account id, e.g. ACC-1001."),
)
def get_account(world: World, account_id: str) -> dict[str, Any]:
    account = dict(world.find("accounts", account_id))
    account["contacts"] = [
        c for c in world.table("contacts") if c.get("account_id") == account_id
    ]
    return account


@tool(
    "crm_list_contacts",
    "List contacts, optionally filtered to a single account.",
    account_id=P.str("Restrict to this account id.", required=False),
)
def list_contacts(world: World, account_id: str | None = None) -> dict[str, Any]:
    rows = world.table("contacts")
    if account_id:
        world.find("accounts", account_id)  # validate; raises if unknown
        rows = [c for c in rows if c.get("account_id") == account_id]
    return {"count": len(rows), "contacts": rows}


@tool(
    "crm_update_account",
    "Update mutable fields on an account. Only the fields you pass are changed.",
    account_id=P.str("Account id to update."),
    health=P.enum(HEALTH, "New health status.", required=False),
    notes=P.str("Free-text note appended to the account.", required=False),
)
def update_account(
    world: World,
    account_id: str,
    health: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    account = world.find("accounts", account_id)
    changed: dict[str, Any] = {}
    if health is not None:
        if health not in HEALTH:
            raise WorldError(f"health must be one of {HEALTH}")
        account["health"] = health
        changed["health"] = health
    if notes is not None:
        account.setdefault("notes", []).append(notes)
        changed["notes"] = notes
    if not changed:
        raise WorldError("update_account requires at least one field to change")
    world.record("crm", "update_account", account_id, **changed)
    return {"ok": True, "account_id": account_id, "changed": changed}
