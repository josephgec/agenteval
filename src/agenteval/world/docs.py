"""Internal wiki: policies, runbooks, and anything the agent writes."""

from __future__ import annotations

from typing import Any

from ..registry import P, tool
from ..state import World, WorldError


@tool(
    "docs.list",
    "List documents in the internal wiki. Returns ids and titles only.",
    prefix=P.str(
        "Only documents whose id starts with this prefix, e.g. 'policy/'.",
        required=False,
    ),
)
def list_docs(world: World, prefix: str | None = None) -> dict[str, Any]:
    rows = world.table("documents")
    if prefix:
        rows = [d for d in rows if d["id"].startswith(prefix)]
    return {
        "count": len(rows),
        "documents": [{"id": d["id"], "title": d.get("title")} for d in rows],
    }


@tool(
    "docs.read",
    "Read one wiki document in full.",
    doc_id=P.str("Document id, e.g. policy/expenses."),
)
def read(world: World, doc_id: str) -> dict[str, Any]:
    return world.find("documents", doc_id)


@tool(
    "docs.write",
    "Create a wiki document, or replace the body of an existing one.",
    doc_id=P.str("Document id / path, e.g. postmortems/2026-07-30-outage."),
    title=P.str("Human-readable title."),
    content=P.str("Full document body in Markdown."),
)
def write(world: World, doc_id: str, title: str, content: str) -> dict[str, Any]:
    if not doc_id.strip():
        raise WorldError("doc_id must not be empty")
    existing = world.maybe_find("documents", doc_id)
    if existing:
        existing.update({"title": title, "content": content, "updated_at": world.today})
        action = "update"
    else:
        world.insert(
            "documents",
            {
                "id": doc_id,
                "title": title,
                "content": content,
                "updated_at": world.today,
                "created_by": "agent",
            },
        )
        action = "create"
    world.record("docs", action, doc_id, title=title, length=len(content))
    return {"ok": True, "doc_id": doc_id, "action": action, "characters": len(content)}
