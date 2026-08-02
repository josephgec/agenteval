"""State assertions for ticket_triage.

Each ticket has one correct (priority, team) pair under the matrix. The
interesting cases:

* TKT-2004 is the precedence test — an enterprise account reporting a
  cross-tenant data exposure. Rule 1 sends it to security, so an agent that
  reads only the tier rules lands on `engineering` and gets the team wrong
  while getting the priority right.
* TKT-2007 is already correct. Touching it is a (small) failure, which is how
  the task measures restraint rather than only completeness.
"""

from agenteval import Trajectory, World, checks

EXPECTED = {
    "TKT-2001": ("P0", "engineering"),  # enterprise, fully down
    "TKT-2002": ("P2", "engineering"),  # mid-market, degraded
    "TKT-2003": ("P3", "billing"),      # billing discrepancy
    "TKT-2004": ("P0", "security"),     # cross-tenant exposure beats tier rules
    "TKT-2005": ("P1", "engineering"),  # enterprise, degraded
    "TKT-2006": ("P2", "support"),      # how-to question
}
ALREADY_CORRECT = "TKT-2007"


def verify(world: World, trajectory: Trajectory):
    c = checks()

    for ticket_id, (priority, team) in EXPECTED.items():
        ticket = world.maybe_find("tickets", ticket_id)
        if ticket is None:
            c.add(f"{ticket_id} still exists", False, detail="record was deleted",
                  weight=2.0)
            continue
        c.equals(f"{ticket_id} priority", ticket.get("priority"), priority,
                 weight=1.0)
        c.equals(f"{ticket_id} team", ticket.get("team"), team, weight=1.0)

    correct = world.maybe_find("tickets", ALREADY_CORRECT)
    touched = [
        m for m in world.mutations_for("tickets") if m.target == ALREADY_CORRECT
    ]
    c.add(
        f"{ALREADY_CORRECT} left untouched (already correctly triaged)",
        correct is not None
        and correct.get("priority") == "P3"
        and correct.get("team") == "billing"
        and not touched,
        detail=(
            f"{len(touched)} write(s) to a ticket already at P3/billing, which "
            "the matrix says to leave alone"
            if correct
            else "ticket is missing"
        ),
        weight=1.0,
    )

    c.add(
        "read the triage matrix",
        any(
            call.input.get("doc_id") == "policy/support-sla"
            for call in trajectory.calls_to("docs.read")
        ),
        detail="never opened policy/support-sla",
        weight=0.5,
    )

    c.add(
        "looked up customer tier in the CRM",
        bool(trajectory.calls_to("crm.get_account"))
        or bool(trajectory.calls_to("crm.search_accounts")),
        detail="never queried the CRM, so tier was guessed from ticket text",
        weight=0.5,
    )

    return c.done()


def _fix(ticket_id: str, priority: str, team: str) -> dict:
    return {
        "tool": "tickets.update",
        "input": {"ticket_id": ticket_id, "priority": priority, "team": team},
    }


GOLD = [
    {"tool": "docs.read", "input": {"doc_id": "policy/support-sla"}},
    {"tool": "tickets.search", "input": {"status": "open"}},
    {"tool": "crm.search_accounts", "input": {}},
    {"tool": "tickets.get", "input": {"ticket_id": "TKT-2001"}},
    {"tool": "tickets.get", "input": {"ticket_id": "TKT-2002"}},
    {"tool": "tickets.get", "input": {"ticket_id": "TKT-2003"}},
    {"tool": "tickets.get", "input": {"ticket_id": "TKT-2004"}},
    {"tool": "tickets.get", "input": {"ticket_id": "TKT-2005"}},
    {"tool": "tickets.get", "input": {"ticket_id": "TKT-2006"}},
    {"tool": "tickets.get", "input": {"ticket_id": "TKT-2007"}},
    {"tool": "crm.get_account", "input": {"account_id": "ACC-1004"}},
    *[_fix(tid, pri, team) for tid, (pri, team) in EXPECTED.items()],
    {
        "say": "Triaged six tickets. TKT-2004 is the one to look at first — "
        "Northgate Logistics is seeing another company's consignment data in "
        "their CSV export, which is a cross-tenant exposure, so it is P0 with "
        "security rather than an engineering performance issue. TKT-2001 is the "
        "other P0: Meridian Health is an enterprise account and fully down. "
        "TKT-2007 was already correctly triaged as P3/billing, so I left it "
        "alone."
    },
]
