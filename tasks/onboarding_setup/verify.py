"""State assertions for onboarding_setup.

The runbook specifies an exact document id and exact section headings, so those
are checkable. Whether the checklist is any *good* is the rubric's job — see
task.yaml. The split is deliberate: structure is cheap and deterministic to
assert, substance is not.

The manager is looked up from the HR directory rather than named in the prompt,
so an agent that assumes rather than checks mails the wrong person.
"""

from agenteval import Trajectory, World, checks

DOC_ID = "onboarding/EMP-045"
MANAGER_EMAIL = "ana.castellanos@northwind.example"
NEW_HIRE_EMAIL = "wei.tanaka@northwind.example"
REQUIRED_SECTIONS = [
    "## Accounts and access",
    "## Equipment",
    "## First week",
    "## Buddy",
]


def verify(world: World, trajectory: Trajectory):
    c = checks()

    doc = world.maybe_find("documents", DOC_ID)
    c.add(
        "checklist created at the runbook's document id",
        doc is not None,
        detail=(
            f"no document at {DOC_ID}; found: "
            f"{[d['id'] for d in world.table('documents') if d.get('created_by')]}"
        ),
        weight=2.0,
    )
    if doc:
        c.contains_all(
            "checklist has all four required sections",
            doc.get("content", ""),
            REQUIRED_SECTIONS,
            weight=2.0,
        )
        c.add(
            "checklist is substantive rather than a stub",
            len(doc.get("content", "")) >= 400,
            detail=f"{len(doc.get('content', ''))} characters",
            weight=0.5,
        )

    it_tickets = [
        t
        for t in world.table("tickets")
        if t["id"].startswith("TKT-NEW") and t.get("team") == "it"
    ]
    c.count("one equipment ticket filed to IT", it_tickets, 1, weight=1.5)
    if it_tickets:
        ticket = it_tickets[0]
        c.equals("equipment ticket priority", ticket.get("priority"), "P2",
                 weight=0.5)
        blob = f"{ticket.get('subject', '')} {ticket.get('body', '')}"
        c.contains_all(
            "equipment ticket names the hire and their start date",
            blob,
            ["Wei Tanaka", "2026-08-11"],
            weight=1.0,
        )

    c.count(
        "hiring manager emailed",
        world.emails_to(MANAGER_EMAIL),
        1,
        weight=1.5,
    )
    c.count(
        "new hire not emailed before their start date",
        world.emails_to(NEW_HIRE_EMAIL),
        0,
        weight=1.0,
    )

    c.add(
        "looked the manager up in the HR directory",
        bool(trajectory.calls_to("hr.get_employee"))
        or bool(trajectory.calls_to("hr.search_employees")),
        detail="never queried HR, so the manager was assumed",
        weight=0.5,
    )

    return c.done()


CHECKLIST = """\
Wei Tanaka joins the Platform team on **2026-08-11**, working remotely from \
Lisbon (WEST, UTC+1). Manager: Ana Castellanos.

## Accounts and access
- Google Workspace account and calendar; add to `eng@` and `platform-team@`
- GitHub organisation invite; write access to `platform-api`, `platform-infra`,
  and `deploy-tooling`
- CI access (Buildkite) and the `platform` deploy role in staging — production
  deploy rights are granted after the first two weeks, per team convention
- Observability: Datadog and the on-call rota in PagerDuty, initially in
  shadow mode
- VPN certificate and SSO enrolment (hardware key posted with the laptop)
- Ticket queue access with the `platform` label

## Equipment
- 16" laptop, 64GB, developer image — shipped to the Lisbon address, so order
  by 2026-08-04 to clear customs before the start date
- External monitor, keyboard, mouse, headset
- Hardware security key (ship separately from the laptop)
- EU power adapters; confirm the home-office stipend with Finance

## First week
- Day 1: welcome call with Ana (30m), laptop and SSO setup with IT
- Day 1-2: platform architecture walkthrough with Dana Whitfield
- Day 2: read `docs/platform/overview` and the runbook index; open a PR
  correcting anything that reads as out of date — the first PR is a
  documentation fix by convention
- Day 3: pair with Dana on a starter issue from the `good-first-issue` label
- Day 4: shadow the on-call handover
- Day 5: retro with Ana on what was missing from this checklist

## Buddy
**Dana Whitfield** (Senior Software Engineer, Platform) — same department, not
in Wei's reporting line. Dana covers the architecture walkthrough on day one and
is the default first port of call for the first month.
"""


GOLD = [
    {"tool": "docs.read", "input": {"doc_id": "policy/onboarding"}},
    {"tool": "hr.get_employee", "input": {"employee_id": "EMP-045"}},
    {"tool": "hr.get_employee", "input": {"employee_id": "EMP-007"}},
    {"tool": "hr.search_employees", "input": {"department": "Engineering"}},
    {
        "tool": "docs.write",
        "input": {
            "doc_id": DOC_ID,
            "title": "Onboarding — Wei Tanaka (EMP-045), Software Engineer, Platform",
            "content": CHECKLIST,
        },
    },
    {
        "tool": "tickets.create",
        "input": {
            "subject": "Equipment for new hire Wei Tanaka — start date 2026-08-11",
            "body": "Wei Tanaka (EMP-045) joins the Platform team on 2026-08-11 "
            "as a Software Engineer, working remotely from Lisbon, Portugal.\n\n"
            "Needed on or before the start date:\n"
            "- 16\" laptop, 64GB, developer image\n"
            "- External monitor, keyboard, mouse, headset\n"
            "- Hardware security key (ship separately from the laptop)\n"
            "- EU power adapters\n\n"
            "This is a remote hire, so everything ships internationally — please "
            "order by 2026-08-04 to allow for shipping and customs. Address is on "
            "the HR record.",
            "team": "it",
            "priority": "P2",
            "assignee": "EMP-018",
        },
    },
    {
        "tool": "email.send",
        "input": {
            "to": [MANAGER_EMAIL],
            "subject": "Wei Tanaka onboarding ready for your review (starts 2026-08-11)",
            "body": "Hi Ana,\n\nWei Tanaka's onboarding is set up ahead of their "
            f"2026-08-11 start date. The checklist is at `{DOC_ID}`.\n\n"
            "What is already done:\n"
            "- Access list drafted for the Platform repos, CI, and staging deploys\n"
            "- Equipment ticket filed with IT, flagged for international shipping "
            "to Lisbon with an order-by date of 2026-08-04\n"
            "- Dana Whitfield pencilled in as buddy — same team, not in Wei's "
            "reporting line\n\n"
            "Two things I need from you by Thursday 2026-08-06:\n"
            "1. Confirm Dana as buddy, or name someone else\n"
            "2. Pick a starter issue for day three, so Wei has something real to "
            "pair on rather than reading documentation for a week\n\n"
            "Production deploy rights are deliberately held back until week "
            "three — say if you would rather that were sooner.\n\nThanks,\n"
            "People Ops",
        },
    },
    {
        "say": "Wei Tanaka's onboarding is ready ahead of the 2026-08-11 start: "
        f"checklist at `{DOC_ID}`, an IT equipment ticket flagged for "
        "international shipping to Lisbon with an order-by date of 2026-08-04, "
        "and Ana Castellanos emailed with the two decisions I need from her. "
        "Dana Whitfield is proposed as buddy — same department, outside Wei's "
        "reporting line."
    },
]
