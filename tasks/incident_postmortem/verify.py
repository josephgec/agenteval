"""State assertions for incident_postmortem.

The structural half is checkable: the document path is specified by the policy
(dated by the day the incident *began*, 2026-07-30, not today), the five
headings are exact, the ticket must close, the account health must move, and
the customer plus the account owner must be told.

Whether the postmortem is *correct* — right root cause, honest blast radius,
owned action items — is the rubric's job. A verifier can confirm the word
"connection pool" appears; only a reader can confirm the analysis is right.
"""

from agenteval import Trajectory, World, checks

DOC_PREFIX = "postmortems/2026-07-30"
CUSTOMER_EMAIL = "a.okonkwo@meridianhealth.example"
ACCOUNT_OWNER_EMAIL = "marcus.reyes@northwind.example"
UNAFFECTED_CUSTOMER = "i.sorensen@northgatelogistics.example"
REQUIRED_SECTIONS = [
    "## Summary",
    "## Impact",
    "## Timeline",
    "## Root cause",
    "## Action items",
]


def _postmortem(world: World) -> dict | None:
    for doc in world.table("documents"):
        if doc["id"].startswith(DOC_PREFIX):
            return doc
    return None


def verify(world: World, trajectory: Trajectory):
    c = checks()

    doc = _postmortem(world)
    c.add(
        "postmortem written at a path dated by the incident date",
        doc is not None,
        detail=(
            f"no document under {DOC_PREFIX}-*; agent-created docs: "
            f"{[d['id'] for d in world.table('documents') if d.get('created_by')]}"
        ),
        weight=2.0,
    )

    if doc:
        content = doc.get("content", "")
        c.contains_all(
            "postmortem has all five required sections",
            content,
            REQUIRED_SECTIONS,
            weight=2.0,
        )
        # Cheap corroboration that the analysis reached the mechanism rather
        # than the symptom. The rubric judges whether it is actually right.
        c.contains_all(
            "root cause names the connection pool exhaustion",
            content,
            ["connection pool"],
            weight=1.0,
        )
        c.add(
            "timeline cites times from the incident record",
            sum(t in content for t in ["06:12", "07:05", "07:11", "07:26"]) >= 3,
            detail="fewer than three of the recorded incident times appear",
            weight=1.0,
        )
        c.add(
            "action item owners are named",
            sum(
                name in content
                for name in ["Dana Whitfield", "Kofi Mensah", "Ana Castellanos"]
            )
            >= 2,
            detail="fewer than two of the debrief's named owners appear",
            weight=1.0,
        )

    ticket = world.maybe_find("tickets", "TKT-3001")
    c.equals(
        "incident ticket closed",
        ticket.get("status") if ticket else None,
        "closed",
        weight=1.5,
    )
    # Order matters: the policy requires the postmortem to exist before the
    # ticket closes. Closing first and writing up afterwards is how postmortems
    # quietly stop happening.
    if ticket and ticket.get("status") == "closed":
        doc_step = next(
            (
                call.step
                for call in trajectory.calls_to("docs_write")
                if str(call.input.get("doc_id", "")).startswith(DOC_PREFIX)
            ),
            None,
        )
        close_step = next(
            (
                call.step
                for call in trajectory.calls_to("tickets_update")
                if call.input.get("status") == "closed"
            ),
            None,
        )
        c.add(
            "postmortem written before the ticket was closed",
            doc_step is not None and close_step is not None and doc_step < close_step,
            detail=f"postmortem at step {doc_step}, close at step {close_step}",
            weight=0.5,
        )

    account = world.maybe_find("accounts", "ACC-1001")
    c.equals(
        "affected account flagged for account-team review",
        account.get("health") if account else None,
        "yellow",
        weight=1.0,
    )

    c.count("customer primary contact emailed", world.emails_to(CUSTOMER_EMAIL), 1,
            weight=1.5)
    c.count(
        "account owner copied",
        world.emails_to(ACCOUNT_OWNER_EMAIL),
        1,
        weight=1.0,
    )
    c.count(
        "unaffected customer not contacted",
        world.emails_to(UNAFFECTED_CUSTOMER),
        0,
        weight=1.5,
    )

    c.add(
        "read the engineer's debrief from the inbox",
        any(
            call.input.get("message_id") == "MSG-4001"
            for call in trajectory.calls_to("email_read")
        ),
        detail="never opened MSG-4001, which carries the action item owners",
        weight=0.5,
    )

    return c.done()


POSTMORTEM = """\
## Summary
On 2026-07-30 a scheduled reporting job exhausted the shared database
connection pool in the eu-west-1 API cluster. Every API request from tenants on
that cluster timed out and returned 503 for 74 minutes. Meridian Health was the
only enterprise tenant affected. The job has been disabled and the connection
leak fixed; the more significant gaps are that no monitoring caught this and
that one tenant's workload can exhaust a pool shared by all tenants on the
cluster.

## Impact
- **Meridian Health** — total API unavailability from 06:12 to 07:26 UTC, 74
  minutes. Clinical staff could not retrieve patient records and two clinics
  fell back to paper records.
- **Northgate Logistics** — not affected. Northgate runs on us-east-1, which
  partitions connection pools per tenant.
- Detection came from the customer, not from monitoring.

## Timeline
All times UTC on 2026-07-30.

| Time | Event |
|---|---|
| 06:00 | Newly deployed scheduled reporting job runs for the first time |
| ~06:12 | Connection pool saturated; customer-facing 503s begin |
| 06:12 | Meridian Health reports total API failure |
| 06:18 | Acknowledged; scope confirmed as the eu-west-1 cluster only |
| 06:34 | API pods found healthy, but all requests blocked waiting on a database connection; pool saturated at its limit of 40 |
| 07:05 | Cause identified: the reporting job deployed 2026-07-29 opens one connection per report row and never returns it to the pool |
| 07:11 | Reporting job disabled; API pods restarted to drain the pool |
| 07:26 | Error rate back to baseline; incident over |

## Root cause
The reporting job deployed on 2026-07-29 acquired a database connection per
report row and never returned it. On its first scheduled run at 06:00 it
exhausted the eu-west-1 connection pool — limit 40 — within about twelve
minutes. Once the pool was empty, API pods stayed healthy but every request
blocked waiting for a connection and eventually returned 503. The 503s were the
symptom; the leak was the cause.

Two conditions turned a leak into a customer-visible outage. There is no alert
on connection pool utilisation, so the pool drained silently and the customer
detected the outage before we did. And eu-west-1 uses a single pool shared
across all tenants, so one background job could exhaust capacity for every
tenant on the cluster — us-east-1 partitions per tenant, which is why Northgate
was unaffected by the same class of failure.

The leak itself is fixed in PR 4412, shipped 2026-07-31.

## Action items
| # | Action | Owner |
|---|---|---|
| 1 | Add alerting on database connection pool utilisation across all clusters, so saturation is detected before it becomes customer-facing | Dana Whitfield |
| 2 | Extend the deploy checklist to require load testing for background and scheduled jobs, not only request-path changes | Kofi Mensah |
| 3 | Scope partitioning the eu-west-1 connection pool per tenant, bringing it in line with us-east-1 | Ana Castellanos |

Item 1 is the highest-value fix: it is what turns this class of failure from a
customer-reported outage into an internal alert.
"""

CUSTOMER_LETTER = """\
Dear Dr. Okonkwo,

Thank you for your patience on Thursday morning, and I am sorry for the
disruption. You asked for a written explanation, and here it is.

**What happened.** A scheduled internal reporting job that we deployed on
Wednesday had a defect: it held on to database connections instead of releasing
them. When the job first ran at 06:00 UTC on Thursday it gradually consumed all
available database connections in the European cluster your account runs on. By
06:12 our API had no connections left to serve your requests, and it began
returning errors. Service was fully restored at 07:26 UTC — a 74 minute
disruption.

**What we got wrong beyond the defect itself.** We did not detect this
ourselves. Your team told us before our monitoring did, and that is not the
standard you should expect from us.

**What we are changing.**
1. We are adding monitoring on the specific resource that was exhausted, so
   this condition raises an alert well before it can affect your service.
2. We are extending our pre-deployment testing requirements to cover background
   jobs of this kind, which were not previously required to be load tested.
3. We are changing how the European cluster allocates database capacity so that
   no single internal process can consume the capacity your service depends on.
   Our US cluster is already configured this way and was unaffected on Thursday.

The defect itself was fixed on Friday and the job remains disabled until the
testing above is complete.

I recognise that two of your clinics fell back to paper records, and I am happy
to walk your board through this in more detail if that would help. Please just
let me know.

With apologies,

Northwind Engineering
"""


GOLD = [
    {"tool": "docs_read", "input": {"doc_id": "policy/postmortems"}},
    {"tool": "tickets_get", "input": {"ticket_id": "TKT-3001"}},
    {"tool": "email_search_inbox", "input": {"query": "outage"}},
    {"tool": "email_read", "input": {"message_id": "MSG-4001"}},
    {"tool": "email_read", "input": {"message_id": "MSG-4002"}},
    {"tool": "crm_get_account", "input": {"account_id": "ACC-1001"}},
    {
        "tool": "docs_write",
        "input": {
            "doc_id": "postmortems/2026-07-30-eu-west-1-connection-pool-exhaustion",
            "title": "Postmortem — eu-west-1 API outage, 2026-07-30",
            "content": POSTMORTEM,
        },
    },
    {
        "tool": "email_send",
        "input": {
            "to": [CUSTOMER_EMAIL],
            "cc": [ACCOUNT_OWNER_EMAIL],
            "subject": "Written explanation — 30 July service disruption",
            "body": CUSTOMER_LETTER,
        },
    },
    {
        "tool": "tickets_comment",
        "input": {
            "ticket_id": "TKT-3001",
            "body": "Postmortem published at "
            "postmortems/2026-07-30-eu-west-1-connection-pool-exhaustion. "
            "Three action items assigned: pool utilisation alerting (Dana), "
            "deploy checklist for background jobs (Kofi), eu-west-1 pool "
            "partitioning (Ana). Customer has been written to. Closing.",
        },
    },
    {"tool": "tickets_update", "input": {"ticket_id": "TKT-3001", "status": "closed"}},
    {
        "tool": "crm_update_account",
        "input": {
            "account_id": "ACC-1001",
            "health": "yellow",
            "notes": "74-minute total API outage on 2026-07-30 (eu-west-1 "
            "connection pool exhaustion). Clinics fell back to paper. Flagged "
            "for weekly account review; renewal is 2026-09-12.",
        },
    },
    {
        "say": "TKT-3001 is closed. The postmortem is at "
        "`postmortems/2026-07-30-eu-west-1-connection-pool-exhaustion`.\n\n"
        "Root cause was a reporting job deployed 2026-07-29 that leaked a "
        "database connection per row and exhausted the shared eu-west-1 pool on "
        "its first run — the 503s were the symptom. Meridian Health was down 74 "
        "minutes; Northgate was unaffected because us-east-1 partitions pools "
        "per tenant. Three action items are assigned to Dana, Kofi and Ana.\n\n"
        "Dr. Okonkwo has the written explanation she asked for, copied to Marcus "
        "as account owner, and I have set Meridian's health to yellow. Worth "
        "flagging that their renewal is 2026-09-12, six weeks out."
    },
]
