"""Selecting what an LLM judge is allowed to see.

A judge shown the whole world grades the seed data as much as the agent's work,
and its scores drift with unrelated changes to the fixture. Each task names the
artifacts its rubric is actually about, and only those are rendered.
"""

from __future__ import annotations

from typing import Any

from ..types import Trajectory
from ..state import World


def _render_email(m: dict[str, Any]) -> str:
    cc = f"\nCc: {', '.join(m['cc'])}" if m.get("cc") else ""
    return (
        f"To: {', '.join(m.get('to', []))}{cc}\n"
        f"Subject: {m.get('subject', '')}\n\n{m.get('body', '')}"
    )


def collect(
    world: World, trajectory: Trajectory, selectors: list[str]
) -> dict[str, str]:
    """Resolve artifact selectors to rendered text.

    Supported selectors:
        final_text          the agent's closing message
        outbox              every email the agent sent
        doc:<id>            one wiki document's body
        docs:<prefix>       every document whose id starts with <prefix>
        new_docs            every document the agent created
        ticket:<id>         one ticket, body and comment thread
        new_tickets         every ticket the agent filed
        expense:<id>        one expense report, including the decision reason
    """
    out: dict[str, str] = {}

    for selector in selectors:
        kind, _, arg = selector.partition(":")

        if kind == "final_text":
            out["final_text"] = trajectory.final_text or "(the agent said nothing)"

        elif kind == "outbox":
            if not world.outbox:
                out["outbox"] = "(no emails were sent)"
            for i, m in enumerate(world.outbox, 1):
                out[f"email_{i}"] = _render_email(m)

        elif kind == "doc":
            doc = world.maybe_find("documents", arg)
            out[f"doc:{arg}"] = (
                f"# {doc.get('title', '')}\n\n{doc.get('content', '')}"
                if doc
                else f"(no document exists at {arg})"
            )

        elif kind == "docs":
            matches = [d for d in world.table("documents") if d["id"].startswith(arg)]
            if not matches:
                out[f"docs:{arg}"] = f"(no documents under {arg})"
            for d in matches:
                out[f"doc:{d['id']}"] = (
                    f"# {d.get('title', '')}\n\n{d.get('content', '')}"
                )

        elif kind == "new_docs":
            created = [
                d for d in world.table("documents") if d.get("created_by") == "agent"
            ]
            if not created:
                out["new_docs"] = "(the agent created no documents)"
            for d in created:
                out[f"doc:{d['id']}"] = (
                    f"# {d.get('title', '')}\n\n{d.get('content', '')}"
                )

        elif kind == "ticket":
            t = world.maybe_find("tickets", arg)
            if not t:
                out[f"ticket:{arg}"] = f"(no ticket {arg})"
            else:
                comments = "\n".join(
                    f"  - {c.get('author')}: {c.get('body')}"
                    for c in t.get("comments", [])
                ) or "  (none)"
                out[f"ticket:{arg}"] = (
                    f"Subject: {t.get('subject')}\nStatus: {t.get('status')} "
                    f"Priority: {t.get('priority')} Team: {t.get('team')}\n\n"
                    f"{t.get('body', '')}\n\nComments:\n{comments}"
                )

        elif kind == "new_tickets":
            created = [
                t for t in world.table("tickets") if t["id"].startswith("TKT-NEW")
            ]
            if not created:
                out["new_tickets"] = "(the agent filed no tickets)"
            for t in created:
                out[f"new_ticket:{t['id']}"] = (
                    f"Subject: {t.get('subject')}\nTeam: {t.get('team')} "
                    f"Priority: {t.get('priority')}\n\n{t.get('body', '')}"
                )

        elif kind == "expense":
            e = world.maybe_find("expenses", arg)
            out[f"expense:{arg}"] = (
                f"Amount: ${e.get('amount_usd')}\nStatus: {e.get('status')}\n"
                f"Decision reason: {e.get('decision_reason', '(none recorded)')}"
                if e
                else f"(no expense {arg})"
            )

        else:
            raise ValueError(
                f"unknown rubric artifact selector {selector!r} — see "
                "agenteval.grading.artifacts.collect for the supported set"
            )

    return out


def render(artifacts: dict[str, str]) -> str:
    if not artifacts:
        return "(no artifacts were produced)"
    return "\n\n".join(
        f"<artifact name=\"{name}\">\n{text}\n</artifact>"
        for name, text in artifacts.items()
    )
