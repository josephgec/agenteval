"""Artifact selectors — what the judge is allowed to see.

A selector that quietly resolves to the wrong thing is worse than one that
crashes: the judge still returns confident-looking scores, just for the wrong
evidence. Two properties matter throughout:

* agent-produced artifacts are distinguished from seeded fixture content, so
  the judge never grades the seed;
* absence is stated explicitly rather than rendered as nothing, because to a
  judge "no artifact" and "no mention of it" read very differently.
"""

import pytest

from agenteval import Trajectory, World
from agenteval.grading import artifacts
from agenteval.world import docs, email, tickets

SEED = {
    "documents": [
        {"id": "policy/x", "title": "Policy X", "content": "seeded policy body"},
        {"id": "policy/y", "title": "Policy Y", "content": "another seeded body"},
        {"id": "runbook/z", "title": "Runbook Z", "content": "runbook body"},
    ],
    "tickets": [
        {"id": "TKT-1", "subject": "Seeded ticket", "body": "seeded ticket body",
         "status": "open", "priority": "P0", "team": "support",
         "comments": [{"author": "Dana", "body": "seeded comment", "at": "d"}]},
    ],
    "expenses": [
        {"id": "EXP-1", "employee_id": "EMP-1", "amount_usd": 4210.5,
         "category": "travel", "status": "submitted"}
    ],
    "employees": [{"id": "EMP-1", "name": "A", "email": "a@x.example"}],
}


@pytest.fixture
def world():
    return World(SEED)


@pytest.fixture
def trajectory():
    t = Trajectory(task_id="t", agent="a")
    t.final_text = "Here is what I did."
    return t


def collect(world, trajectory, *selectors):
    return artifacts.collect(world, trajectory, list(selectors))


# --------------------------------------------------------------------------- #
# Individual selectors
# --------------------------------------------------------------------------- #


def test_final_text(world, trajectory):
    assert collect(world, trajectory, "final_text")["final_text"] == (
        "Here is what I did."
    )


def test_final_text_says_so_when_the_agent_said_nothing(world):
    empty = Trajectory(task_id="t", agent="a")
    assert "said nothing" in collect(world, empty, "final_text")["final_text"]


def test_outbox_renders_headers_and_body(world, trajectory):
    email.send(world, to=["a@x.example"], subject="Subject line",
               body="Body text", cc=["b@y.example"])
    rendered = collect(world, trajectory, "outbox")["email_1"]
    assert "To: a@x.example" in rendered
    assert "Cc: b@y.example" in rendered
    assert "Subject: Subject line" in rendered
    assert "Body text" in rendered


def test_outbox_numbers_each_message_separately(world, trajectory):
    email.send(world, to=["a@x.example"], subject="one", body="b")
    email.send(world, to=["b@x.example"], subject="two", body="b")
    result = collect(world, trajectory, "outbox")
    assert set(result) == {"email_1", "email_2"}


def test_outbox_states_that_nothing_was_sent(world, trajectory):
    assert "no emails were sent" in collect(world, trajectory, "outbox")["outbox"]


def test_doc_selector_returns_title_and_body(world, trajectory):
    rendered = collect(world, trajectory, "doc:policy/x")["doc:policy/x"]
    assert "# Policy X" in rendered and "seeded policy body" in rendered


def test_doc_selector_states_absence(world, trajectory):
    rendered = collect(world, trajectory, "doc:nope/1")["doc:nope/1"]
    assert "no document exists at nope/1" in rendered


def test_docs_prefix_selects_only_matching_ids(world, trajectory):
    result = collect(world, trajectory, "docs:policy/")
    assert set(result) == {"doc:policy/x", "doc:policy/y"}


def test_docs_prefix_states_absence(world, trajectory):
    assert "no documents under absent/" in (
        collect(world, trajectory, "docs:absent/")["docs:absent/"]
    )


def test_new_docs_excludes_seeded_documents(world, trajectory):
    """Otherwise the judge grades the fixture the task author wrote."""
    docs.write(world, "postmortems/p1", "Postmortem", "agent-written body")
    result = collect(world, trajectory, "new_docs")
    assert set(result) == {"doc:postmortems/p1"}
    assert "agent-written body" in result["doc:postmortems/p1"]


def test_new_docs_states_absence(world, trajectory):
    assert "created no documents" in collect(world, trajectory, "new_docs")["new_docs"]


def test_ticket_selector_includes_the_comment_thread(world, trajectory):
    tickets.comment(world, "TKT-1", "agent comment")
    rendered = collect(world, trajectory, "ticket:TKT-1")["ticket:TKT-1"]
    assert "Seeded ticket" in rendered
    assert "Status: open" in rendered and "Priority: P0" in rendered
    assert "seeded comment" in rendered and "agent comment" in rendered


def test_ticket_selector_renders_an_empty_thread(world, trajectory):
    world.table("tickets")[0]["comments"] = []
    assert "(none)" in collect(world, trajectory, "ticket:TKT-1")["ticket:TKT-1"]


def test_ticket_selector_states_absence(world, trajectory):
    assert "no ticket TKT-99" in collect(world, trajectory, "ticket:TKT-99")[
        "ticket:TKT-99"
    ]


def test_new_tickets_excludes_seeded_tickets(world, trajectory):
    tickets.create(world, "Agent filed", "body", "it", "P2")
    result = collect(world, trajectory, "new_tickets")
    assert len(result) == 1
    [rendered] = result.values()
    assert "Agent filed" in rendered
    assert "Seeded ticket" not in rendered


def test_new_tickets_states_absence(world, trajectory):
    assert "filed no tickets" in collect(world, trajectory, "new_tickets")[
        "new_tickets"
    ]


def test_expense_selector_exposes_the_decision_reason(world, trajectory):
    from agenteval.world import expenses

    expenses.decide(world, "EXP-1", "reject", "no receipt attached")
    rendered = collect(world, trajectory, "expense:EXP-1")["expense:EXP-1"]
    assert "4210.5" in rendered
    assert "rejected" in rendered
    assert "no receipt attached" in rendered


def test_expense_selector_marks_an_undecided_report(world, trajectory):
    rendered = collect(world, trajectory, "expense:EXP-1")["expense:EXP-1"]
    assert "(none recorded)" in rendered


def test_expense_selector_states_absence(world, trajectory):
    assert "no expense EXP-99" in collect(world, trajectory, "expense:EXP-99")[
        "expense:EXP-99"
    ]


# --------------------------------------------------------------------------- #
# Composition and misuse
# --------------------------------------------------------------------------- #


def test_selectors_compose(world, trajectory):
    email.send(world, to=["a@x.example"], subject="s", body="b")
    result = collect(world, trajectory, "final_text", "outbox", "doc:policy/x")
    assert set(result) == {"final_text", "email_1", "doc:policy/x"}


def test_an_unknown_selector_fails_loudly():
    """Task authoring typos must not silently show the judge nothing."""
    with pytest.raises(ValueError, match="unknown rubric artifact selector"):
        artifacts.collect(World(SEED), Trajectory("t", "a"), ["outbx"])


def test_render_wraps_each_artifact_with_its_name():
    rendered = artifacts.render({"email_1": "hello", "final_text": "done"})
    assert '<artifact name="email_1">' in rendered
    assert '<artifact name="final_text">' in rendered
    assert "hello" in rendered and "done" in rendered


def test_render_states_when_nothing_was_produced():
    assert "no artifacts" in artifacts.render({})
