"""The simulated systems: mutations, audit log, and error surfaces."""

import pytest

from agenteval import World
from agenteval.state import WorldError
from agenteval.world import crm, docs, email, expenses, tickets

SEED = {
    "today": "2026-08-01",
    "accounts": [
        {
            "id": "ACC-1",
            "name": "Acme Corp",
            "tier": "enterprise",
            "arr_usd": 250000,
            "renewal_date": "2026-09-01",
            "health": "green",
        },
        {
            "id": "ACC-2",
            "name": "Beta Ltd",
            "tier": "smb",
            "arr_usd": 12000,
            "renewal_date": "2026-11-01",
            "health": "green",
        },
    ],
    "contacts": [
        {"id": "CON-1", "account_id": "ACC-1", "name": "Ada", "email": "ada@acme.example"}
    ],
    "employees": [
        {"id": "EMP-1", "name": "Boss", "email": "boss@co.example", "manager_id": None},
        {"id": "EMP-2", "name": "Report", "email": "rep@co.example",
         "manager_id": "EMP-1"},
    ],
    "tickets": [
        {"id": "TKT-1", "subject": "Down", "body": "b", "status": "open",
         "priority": "P3", "team": "support", "assignee": None,
         "account_id": "ACC-1", "created_at": "2026-08-01", "comments": []}
    ],
    "documents": [{"id": "policy/x", "title": "X", "content": "body"}],
    "expenses": [
        {"id": "EXP-1", "employee_id": "EMP-2", "amount_usd": 100.0,
         "category": "meals", "status": "submitted"}
    ],
}


@pytest.fixture
def world():
    return World(SEED)


def test_seed_is_deep_copied(world):
    """Two worlds from one seed must not share mutable state."""
    crm.update_account(world, "ACC-1", health="red")
    assert World(SEED).find("accounts", "ACC-1")["health"] == "green"


def test_unknown_seed_collection_is_rejected():
    with pytest.raises(WorldError, match="unknown collections"):
        World({"acounts": []})  # typo


def test_missing_record_raises_with_known_ids(world):
    with pytest.raises(WorldError, match="ACC-999"):
        world.find("accounts", "ACC-999")


def test_account_search_filters_combine(world):
    both = crm.search_accounts(world, min_arr_usd=100000, renewal_before="2026-10-01")
    assert [a["id"] for a in both["accounts"]] == ["ACC-1"]
    # ACC-2 renews inside no window and is under the ARR bar
    assert crm.search_accounts(world, min_arr_usd=100000)["count"] == 1
    assert crm.search_accounts(world, renewal_before="2026-10-01")["count"] == 1


def test_ticket_update_records_mutation(world):
    tickets.update(world, "TKT-1", priority="P0", team="security")
    ticket = world.find("tickets", "TKT-1")
    assert (ticket["priority"], ticket["team"]) == ("P0", "security")
    [mutation] = world.mutations_for("tickets", "update")
    assert mutation.target == "TKT-1"
    assert mutation.payload == {"priority": "P0", "team": "security"}


def test_ticket_update_with_no_fields_is_an_error(world):
    with pytest.raises(WorldError, match="at least one field"):
        tickets.update(world, "TKT-1")


def test_ticket_create_validates_foreign_keys(world):
    with pytest.raises(WorldError, match="ACC-99"):
        tickets.create(world, "s", "b", "it", "P2", account_id="ACC-99")


def test_email_send_validates_addresses(world):
    with pytest.raises(WorldError, match="not valid email"):
        email.send(world, to=["not-an-address"], subject="s", body="b")
    with pytest.raises(WorldError, match="at least one recipient"):
        email.send(world, to=[], subject="s", body="b")


def test_emails_to_matches_cc_case_insensitively(world):
    email.send(world, to=["a@x.example"], subject="s", body="b",
               cc=["Boss@CO.example"])
    assert len(world.emails_to("boss@co.example")) == 1


def test_docs_write_creates_then_updates(world):
    assert docs.write(world, "n/1", "T", "c")["action"] == "create"
    assert docs.write(world, "n/1", "T2", "c2")["action"] == "update"
    assert world.find("documents", "n/1")["content"] == "c2"
    # created_by marks agent output, which the artifact selector relies on
    assert world.find("documents", "n/1")["created_by"] == "agent"


def test_expense_decide_is_permissive_by_design(world):
    """Policy lives in the task, not the environment.

    If the world refused an over-limit approval, an agent could not fail the
    policy check and the task would measure nothing.
    """
    expenses.decide(world, "EXP-1", "approve", "because")
    assert world.find("expenses", "EXP-1")["status"] == "approved"


def test_expense_escalate_requires_a_target(world):
    with pytest.raises(WorldError, match="escalate requires"):
        expenses.decide(world, "EXP-1", "escalate", "r")
    with pytest.raises(WorldError, match="EMP-99"):
        expenses.decide(world, "EXP-1", "escalate", "r", escalate_to="EMP-99")


def test_broadcast_is_recorded_as_a_broadcast(world):
    email.broadcast_all_staff(world, "s", "b")
    assert world.outbox[-1]["broadcast"] is True
    assert len(world.outbox[-1]["to"]) == 2
