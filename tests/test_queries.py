"""Search and lookup across the simulated systems.

These are the surfaces an agent reasons from. A filter that silently returns
the wrong rows doesn't crash anything — it makes the agent draw a wrong
conclusion from correct reasoning, and the eval scores a harness bug as a model
failure. Each filter is exercised alone and in combination.
"""

import pytest

from agenteval import World
from agenteval.state import WorldError
from agenteval.world import crm, docs, expenses, hr, tickets

SEED = {
    "accounts": [
        {"id": "ACC-1", "name": "Meridian Health", "tier": "enterprise",
         "arr_usd": 480000, "renewal_date": "2026-09-12", "health": "green"},
        {"id": "ACC-2", "name": "Volta Robotics", "tier": "mid_market",
         "arr_usd": 96000, "renewal_date": "2026-08-28", "health": "green"},
        {"id": "ACC-3", "name": "Kestrel Legal", "tier": "smb",
         "arr_usd": 18000, "renewal_date": "2026-12-30", "health": "yellow"},
    ],
    "contacts": [
        {"id": "CON-1", "account_id": "ACC-1", "name": "Amara", "email": "a@m.example"},
        {"id": "CON-2", "account_id": "ACC-2", "name": "Sofia", "email": "s@v.example"},
    ],
    "tickets": [
        {"id": "TKT-1", "subject": "Down", "body": "b1", "status": "open",
         "priority": "P0", "team": "engineering", "assignee": None,
         "account_id": "ACC-1", "comments": []},
        {"id": "TKT-2", "subject": "Slow", "body": "b2", "status": "open",
         "priority": "P2", "team": "support", "assignee": "EMP-1",
         "account_id": "ACC-2", "comments": []},
        {"id": "TKT-3", "subject": "Bill", "body": "b3", "status": "closed",
         "priority": "P3", "team": "billing", "assignee": None,
         "account_id": "ACC-1", "comments": []},
    ],
    "employees": [
        {"id": "EMP-1", "name": "Ana Castellanos", "email": "ana@co.example",
         "title": "Director of Engineering", "department": "Engineering",
         "manager_id": None},
        {"id": "EMP-2", "name": "Dana Whitfield", "email": "dana@co.example",
         "title": "Senior Software Engineer", "department": "Engineering",
         "manager_id": "EMP-1"},
        {"id": "EMP-3", "name": "Tomas Oyelaran", "email": "tomas@co.example",
         "title": "Support Specialist", "department": "Support",
         "manager_id": None},
    ],
    "documents": [
        {"id": "policy/a", "title": "A", "content": "x"},
        {"id": "policy/b", "title": "B", "content": "y"},
        {"id": "runbook/c", "title": "C", "content": "z"},
    ],
    "expenses": [
        {"id": "EXP-1", "employee_id": "EMP-2", "amount_usd": 340.0,
         "category": "software", "status": "submitted"},
        {"id": "EXP-2", "employee_id": "EMP-2", "amount_usd": 4210.5,
         "category": "travel", "status": "submitted"},
        {"id": "EXP-3", "employee_id": "EMP-3", "amount_usd": 920.0,
         "category": "meals", "status": "approved"},
    ],
}


@pytest.fixture
def world():
    return World(SEED)


def ids(result, key):
    return sorted(row["id"] for row in result[key])


# --------------------------------------------------------------------------- #
# CRM
# --------------------------------------------------------------------------- #


def test_account_name_search_is_case_insensitive_and_partial(world):
    assert ids(crm.search_accounts(world, query="meridian"), "accounts") == ["ACC-1"]
    assert ids(crm.search_accounts(world, query="ROBOT"), "accounts") == ["ACC-2"]


def test_account_tier_filter(world):
    assert ids(crm.search_accounts(world, tier="enterprise"), "accounts") == ["ACC-1"]


def test_renewal_before_is_exclusive_of_the_boundary(world):
    """A task saying 'renews before 2026-10-01' must not include 2026-10-01."""
    world.table("accounts")[2]["renewal_date"] = "2026-10-01"
    assert ids(crm.search_accounts(world, renewal_before="2026-10-01"),
               "accounts") == ["ACC-1", "ACC-2"]


def test_accounts_without_a_renewal_date_are_excluded_from_date_filters(world):
    del world.table("accounts")[0]["renewal_date"]
    assert "ACC-1" not in ids(
        crm.search_accounts(world, renewal_before="2027-01-01"), "accounts"
    )


def test_min_arr_is_inclusive(world):
    assert ids(crm.search_accounts(world, min_arr_usd=96000),
               "accounts") == ["ACC-1", "ACC-2"]


def test_account_search_with_no_filters_returns_everything(world):
    assert crm.search_accounts(world)["count"] == 3


def test_account_filters_combine_as_and(world):
    """The renewal_outreach task depends on this: each excluded account fails
    exactly one condition."""
    result = crm.search_accounts(world, min_arr_usd=100000,
                                 renewal_before="2026-10-01")
    assert ids(result, "accounts") == ["ACC-1"]


def test_get_account_attaches_its_contacts(world):
    account = crm.get_account(world, "ACC-1")
    assert [c["id"] for c in account["contacts"]] == ["CON-1"]


def test_list_contacts_validates_the_account_id(world):
    assert crm.list_contacts(world, account_id="ACC-1")["count"] == 1
    with pytest.raises(WorldError, match="ACC-99"):
        crm.list_contacts(world, account_id="ACC-99")


def test_update_account_appends_notes_rather_than_replacing(world):
    crm.update_account(world, "ACC-1", notes="first")
    crm.update_account(world, "ACC-1", notes="second")
    assert world.find("accounts", "ACC-1")["notes"] == ["first", "second"]


def test_update_account_rejects_an_invalid_health(world):
    with pytest.raises(WorldError, match="health must be one of"):
        crm.update_account(world, "ACC-1", health="excellent")


# --------------------------------------------------------------------------- #
# Tickets
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"status": "open"}, ["TKT-1", "TKT-2"]),
        ({"priority": "P0"}, ["TKT-1"]),
        ({"team": "billing"}, ["TKT-3"]),
        ({"account_id": "ACC-1"}, ["TKT-1", "TKT-3"]),
        ({"unassigned_only": True}, ["TKT-1", "TKT-3"]),
        ({"status": "open", "unassigned_only": True}, ["TKT-1"]),
        ({"status": "open", "team": "billing"}, []),
    ],
)
def test_ticket_search_filters(world, kwargs, expected):
    assert ids(tickets.search(world, **kwargs), "tickets") == expected


def test_unassigned_only_false_does_not_filter(world):
    """A falsy flag must mean 'no filter', not 'only assigned'."""
    assert len(tickets.search(world, unassigned_only=False)["tickets"]) == 3


def test_created_tickets_are_findable_and_carry_defaults(world):
    result = tickets.create(world, "New", "body", "it", "P2")
    ticket = world.find("tickets", result["ticket"]["id"])
    assert ticket["status"] == "open"
    assert ticket["assignee"] is None
    assert ticket["id"].startswith("TKT-NEW")


def test_created_ticket_ids_are_unique(world):
    first = tickets.create(world, "a", "b", "it", "P2")["ticket"]["id"]
    second = tickets.create(world, "c", "d", "it", "P2")["ticket"]["id"]
    assert first != second


def test_create_validates_the_assignee(world):
    with pytest.raises(WorldError, match="EMP-99"):
        tickets.create(world, "s", "b", "it", "P2", assignee="EMP-99")


def test_update_rejects_an_unknown_assignee(world):
    with pytest.raises(WorldError, match="EMP-99"):
        tickets.update(world, "TKT-1", assignee="EMP-99")


def test_comments_accumulate_in_order(world):
    tickets.comment(world, "TKT-1", "first")
    tickets.comment(world, "TKT-1", "second")
    bodies = [c["body"] for c in world.find("tickets", "TKT-1")["comments"]]
    assert bodies == ["first", "second"]


# --------------------------------------------------------------------------- #
# HR
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"query": "dana"}, ["EMP-2"]),
        ({"query": "tomas@co.example"}, ["EMP-3"]),   # matches on email
        ({"query": "director"}, ["EMP-1"]),           # matches on title
        ({"department": "Engineering"}, ["EMP-1", "EMP-2"]),
        ({"manager_id": "EMP-1"}, ["EMP-2"]),
        ({"department": "Support", "query": "specialist"}, ["EMP-3"]),
    ],
)
def test_employee_search_filters(world, kwargs, expected):
    assert ids(hr.search_employees(world, **kwargs), "employees") == expected


def test_get_employee_resolves_the_manager(world):
    employee = hr.get_employee(world, "EMP-2")
    assert employee["manager"]["name"] == "Ana Castellanos"


def test_get_employee_omits_manager_at_the_top_of_the_tree(world):
    assert "manager" not in hr.get_employee(world, "EMP-1")


def test_get_employee_tolerates_a_dangling_manager_id(world):
    world.find("employees", "EMP-2")["manager_id"] = "EMP-GONE"
    assert "manager" not in hr.get_employee(world, "EMP-2")


# --------------------------------------------------------------------------- #
# Docs and expenses
# --------------------------------------------------------------------------- #


def test_docs_list_prefix_filter(world):
    assert ids(docs.list_docs(world, prefix="policy/"), "documents") == [
        "policy/a", "policy/b"
    ]
    assert docs.list_docs(world)["count"] == 3


def test_docs_list_returns_titles_without_bodies(world):
    """Listing the whole wiki with bodies would flood the context window."""
    [entry, *_] = docs.list_docs(world)["documents"]
    assert set(entry) == {"id", "title"}


def test_docs_write_rejects_an_empty_id(world):
    with pytest.raises(WorldError, match="must not be empty"):
        docs.write(world, "   ", "T", "c")


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"status": "submitted"}, ["EXP-1", "EXP-2"]),
        ({"employee_id": "EMP-2"}, ["EXP-1", "EXP-2"]),
        ({"min_amount_usd": 1000}, ["EXP-2"]),
        ({"status": "submitted", "min_amount_usd": 1000}, ["EXP-2"]),
        ({"employee_id": "EMP-3", "status": "submitted"}, []),
    ],
)
def test_expense_search_filters(world, kwargs, expected):
    assert ids(expenses.search(world, **kwargs), "expenses") == expected


def test_expense_decision_records_reason_and_date(world):
    expenses.decide(world, "EXP-1", "reject", "no receipt")
    expense = world.find("expenses", "EXP-1")
    assert expense["status"] == "rejected"
    assert expense["decision_reason"] == "no receipt"
    assert expense["decided_at"] == world.today


def test_escalation_records_the_target(world):
    expenses.decide(world, "EXP-2", "escalate", "over threshold", escalate_to="EMP-1")
    assert world.find("expenses", "EXP-2")["escalated_to"] == "EMP-1"


def test_an_invalid_decision_is_rejected(world):
    with pytest.raises(WorldError, match="decision must be one of"):
        expenses.decide(world, "EXP-1", "maybe", "r")


# --------------------------------------------------------------------------- #
# Remaining guards
# --------------------------------------------------------------------------- #


def test_inbox_search_by_sender(world):
    from agenteval.world import email

    world.table("inbox").extend(
        [
            {"id": "MSG-1", "from": "Dana@co.example", "to": [], "subject": "s",
             "body": "b", "received_at": "d"},
            {"id": "MSG-2", "from": "customer@x.example", "to": [], "subject": "s",
             "body": "b", "received_at": "d"},
        ]
    )
    assert ids(email.search_inbox(world, sender="dana"), "messages") == ["MSG-1"]
    assert email.search_inbox(world, sender="nobody")["count"] == 0


def test_inbox_previews_are_truncated(world):
    """Full bodies in a search result would flood the context window."""
    from agenteval.world import email

    world.table("inbox").append(
        {"id": "MSG-1", "from": "a@x.example", "to": [], "subject": "s",
         "body": "x" * 500, "received_at": "d"}
    )
    [preview] = email.search_inbox(world)["messages"]
    assert len(preview["preview"]) < 200
    assert preview["preview"].endswith("…")


def test_reading_a_message_marks_it_read(world):
    from agenteval.world import email

    world.table("inbox").append(
        {"id": "MSG-1", "from": "a@x.example", "to": [], "subject": "s",
         "body": "full body", "received_at": "d", "read": False}
    )
    assert email.read(world, "MSG-1")["body"] == "full body"
    assert world.find("inbox", "MSG-1")["read"] is True


def test_ticket_update_rejects_an_invalid_enum_value(world):
    with pytest.raises(WorldError, match="priority must be one of"):
        tickets.update(world, "TKT-1", priority="URGENT")


def test_ticket_update_can_set_only_the_assignee(world):
    tickets.update(world, "TKT-1", assignee="EMP-1")
    assert world.find("tickets", "TKT-1")["assignee"] == "EMP-1"
    [mutation] = world.mutations_for("tickets", "update")
    assert mutation.payload == {"assignee": "EMP-1"}


def test_update_account_with_no_fields_is_an_error(world):
    with pytest.raises(WorldError, match="at least one field"):
        crm.update_account(world, "ACC-1")
