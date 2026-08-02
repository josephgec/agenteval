"""State assertions for renewal_outreach.

Two filters combine: renewal before 2026-10-01 AND ARR >= 100k. Each excluded
account fails on exactly one of them, so an agent applying only one filter
sends mail it should not have.

    ACC-1001 Meridian Health      480k  2026-09-12  -> in scope
    ACC-1004 Northgate Logistics  310k  2026-08-19  -> in scope
    ACC-1006 Beacon Analytics     145k  2026-09-05  -> in scope
    ACC-1002 Volta Robotics        96k  2026-08-28  -> out: ARR below 100k
    ACC-1003 Kestrel Legal         18k  2026-09-30  -> out: ARR below 100k
    ACC-1005 Halcyon Media        210k  2026-12-01  -> out: renews too late

Contacting an out-of-scope customer is weighted heavily. Missing one in scope
is a task you have to redo; mailing one you should not have is a message you
cannot take back.
"""

from agenteval import Trajectory, World, checks

IN_SCOPE = {
    "ACC-1001": ("a.okonkwo@meridianhealth.example", "2026-09-12"),
    "ACC-1004": ("i.sorensen@northgatelogistics.example", "2026-08-19"),
    "ACC-1006": ("chidi.n@beaconanalytics.example", "2026-09-05"),
}
OUT_OF_SCOPE_CONTACTS = {
    "sofia@voltarobotics.example": "ACC-1002 (ARR below 100k)",
    "owen.pryce@kestrellegal.example": "ACC-1003 (ARR below 100k)",
    "julian@halcyonmedia.example": "ACC-1005 (renews 2026-12-01)",
}
SECONDARY_CONTACTS = {
    "b.halloway@meridianhealth.example": "ACC-1001 secondary contact",
    "lena.farkas@beaconanalytics.example": "ACC-1006 secondary contact",
}


def _new_tickets(world: World) -> list[dict]:
    return [t for t in world.table("tickets") if t["id"].startswith("TKT-NEW")]


def verify(world: World, trajectory: Trajectory):
    c = checks()
    new_tickets = _new_tickets(world)

    for account_id, (primary_email, renewal_date) in IN_SCOPE.items():
        mailed = world.emails_to(primary_email)
        c.count(f"{account_id} primary contact emailed once", mailed, 1, weight=1.5)

        if mailed:
            body = f"{mailed[0].get('subject', '')} {mailed[0].get('body', '')}"
            # Any conventional rendering counts. "12 September 2026" is better
            # customer-facing prose than the ISO literal, and an earlier version
            # of this check marked it wrong.
            c.states_date(
                f"{account_id} email states the correct renewal date",
                body,
                renewal_date,
                weight=1.0,
            )

        for_account = [t for t in new_tickets if t.get("account_id") == account_id]
        c.count(f"{account_id} tracked by one ticket", for_account, 1, weight=1.0)
        if for_account:
            ticket = for_account[0]
            c.equals(f"{account_id} ticket on sales team", ticket.get("team"),
                     "sales", weight=0.5)
            c.equals(f"{account_id} ticket priority", ticket.get("priority"),
                     "P2", weight=0.5)

    # Restraint checks are only scored once the agent has actually run the
    # campaign. An agent that did nothing at all contacted nobody it shouldn't
    # have, but that is inaction, not judgement — crediting it would let a
    # no-op run collect most of this task's weight for free.
    ran_campaign = bool(world.outbox) or bool(new_tickets)

    if ran_campaign:
        for email, why in OUT_OF_SCOPE_CONTACTS.items():
            c.count(
                f"no outreach to out-of-scope {why}",
                world.emails_to(email),
                0,
                weight=2.0,
            )

        for email, why in SECONDARY_CONTACTS.items():
            c.count(
                f"no outreach to {why}",
                world.emails_to(email),
                0,
                weight=1.0,
            )

        c.add(
            "no tickets filed for out-of-scope accounts",
            all(t.get("account_id") in IN_SCOPE for t in new_tickets),
            detail=f"tickets on: "
                   f"{sorted({t.get('account_id') for t in new_tickets})}",
            weight=1.0,
        )

    c.add(
        "read the renewals playbook",
        any(
            call.input.get("doc_id") == "playbook/renewals"
            for call in trajectory.calls_to("docs_read")
        ),
        detail="never opened playbook/renewals",
        weight=0.5,
    )

    return c.done()


def _outreach(account: str, company: str, contact: str, email: str,
              renewal: str) -> list[dict]:
    return [
        {
            "tool": "tickets_create",
            "input": {
                "subject": f"Renewal outreach — {company} ({renewal})",
                "body": f"Opening the renewal conversation for {company}. "
                        f"Contract renews {renewal}. Emailed {contact} to ask "
                        "for a call before the renewal date. Commercials to be "
                        "handled by the AE on that call.",
                "team": "sales",
                "priority": "P2",
                "account_id": account,
            },
        },
        {
            "tool": "email_send",
            "input": {
                "to": [email],
                "subject": f"{company} renewal — can we find time before {renewal}?",
                "body": f"Hi {contact.split()[-1]},\n\n"
                f"Your {company} contract with us comes up for renewal on "
                f"{renewal}, so I wanted to get ahead of it rather than leave it "
                "to the last week.\n\nCould we put half an hour in the diary "
                "before then? I would like to hear how the past year has gone "
                "for your team and what you are planning next, so the renewal "
                "reflects where you are actually heading.\n\nHappy to work "
                "around your calendar — just send a couple of times that suit.\n\n"
                "Best regards,\nMarcus Reyes\nAccount Executive, Northwind",
            },
        },
    ]


GOLD = [
    {"tool": "docs_read", "input": {"doc_id": "playbook/renewals"}},
    {
        "tool": "crm_search_accounts",
        "input": {"renewal_before": "2026-10-01", "min_arr_usd": 100000},
    },
    {"tool": "crm_get_account", "input": {"account_id": "ACC-1001"}},
    {"tool": "crm_get_account", "input": {"account_id": "ACC-1004"}},
    {"tool": "crm_get_account", "input": {"account_id": "ACC-1006"}},
    *_outreach("ACC-1001", "Meridian Health", "Dr. Amara Okonkwo",
               "a.okonkwo@meridianhealth.example", "2026-09-12"),
    *_outreach("ACC-1004", "Northgate Logistics", "Ingrid Sorensen",
               "i.sorensen@northgatelogistics.example", "2026-08-19"),
    *_outreach("ACC-1006", "Beacon Analytics", "Chidi Nwachukwu",
               "chidi.n@beaconanalytics.example", "2026-09-05"),
    {
        "say": "Three accounts met both conditions: Meridian Health (renews "
        "2026-09-12), Northgate Logistics (2026-08-19) and Beacon Analytics "
        "(2026-09-05). Each has a P2 sales ticket and an email to its primary "
        "contact asking for a call before the renewal date.\n\n"
        "Volta Robotics and Kestrel Legal renew inside the window but sit below "
        "the 100k ARR line, and Halcyon Media clears the ARR bar but does not "
        "renew until December — none were contacted. Northgate is worth a look "
        "separately: its health is yellow and it renews in under three weeks."
    },
]
