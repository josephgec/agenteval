"""ToolSession: schema generation, dispatch, blocking, and the step budget.

These are the harness guarantees that have to hold regardless of what the agent
under test does, so they get tested directly rather than through an agent.
"""

import json
import re

import pytest

from agenteval import REGISTRY, TaskSpec, ToolSession, Trajectory, World, all_tools
from agenteval.registry import BudgetExceeded

SEED = {
    "tickets": [
        {"id": "TKT-1", "subject": "s", "body": "b", "status": "open",
         "priority": "P3", "team": "support", "comments": []}
    ],
    "employees": [{"id": "EMP-1", "name": "A", "email": "a@x.example"}],
}


def make_session(**task_kwargs):
    spec = TaskSpec(id="t", prompt="p", seed=SEED, **task_kwargs)
    world = World(spec.seed)
    trajectory = Trajectory(task_id="t", agent="test")
    return world, trajectory, ToolSession(world, spec, trajectory)


# -- schema ---------------------------------------------------------------- #


def test_every_tool_has_a_well_formed_schema():
    for definition in all_tools():
        schema = definition.schema
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert definition.description.strip(), f"{definition.name} has no description"
        for name in schema["required"]:
            assert name in schema["properties"], f"{definition.name}.{name}"
        for name, prop in schema["properties"].items():
            assert prop.get("description"), f"{definition.name}.{name} undocumented"


def test_tool_order_is_stable():
    """An unsorted tool block silently invalidates the prompt cache each turn."""
    assert [t.name for t in all_tools()] == sorted(REGISTRY)


#: The Anthropic API constrains tool names to this, and OpenAI's function-name
#: rule is the same shape. Ollama is the permissive outlier — it accepts dots,
#: which is why an invalid scheme can pass a local run and then 400 on the
#: first paid one.
API_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


@pytest.mark.parametrize("definition", all_tools(), ids=lambda d: d.name)
def test_tool_names_are_valid_for_the_api(definition):
    assert API_TOOL_NAME.match(definition.name), (
        f"{definition.name!r} is rejected by the Messages API "
        "(tools.N.custom.name must match ^[a-zA-Z0-9_-]{1,128}$). "
        "Dots are the usual culprit."
    )


# -- dispatch -------------------------------------------------------------- #


def test_successful_call_is_recorded():
    world, trajectory, session = make_session()
    text, is_error = session.call("tickets_get", {"ticket_id": "TKT-1"})
    assert not is_error
    assert json.loads(text)["subject"] == "s"
    [call] = trajectory.calls
    assert (call.name, call.step, call.blocked_reason) == ("tickets_get", 1, None)


def test_world_errors_come_back_as_tool_errors_not_exceptions():
    _, trajectory, session = make_session()
    text, is_error = session.call("tickets_get", {"ticket_id": "TKT-404"})
    assert is_error and "TKT-404" in text
    # Recorded, but not "blocked" — the agent made a legitimate call that failed
    assert trajectory.calls[0].blocked_reason is None


def test_missing_and_unexpected_arguments_are_rejected():
    _, trajectory, session = make_session()
    _, err1 = session.call("tickets_get", {})
    _, err2 = session.call("tickets_get", {"ticket_id": "TKT-1", "nope": 1})
    assert err1 and err2
    assert [c.blocked_reason for c in trajectory.calls] == ["bad_args", "bad_args"]


def test_unknown_tool_is_rejected_without_crashing():
    _, trajectory, session = make_session()
    text, is_error = session.call("tickets.explode", {})
    assert is_error and "Unknown tool" in text
    assert trajectory.calls[0].blocked_reason == "unknown"


# -- exposure and blocking -------------------------------------------------- #


def test_allowed_tools_restricts_the_exposed_set():
    _, _, session = make_session(allowed_tools=["tickets_get", "tickets_update"])
    assert [t.name for t in session.tools] == ["tickets_get", "tickets_update"]


def test_forbidden_tools_stay_visible_but_are_blocked():
    """Exposure is the point: we measure reaching, not just succeeding."""
    world, trajectory, session = make_session(
        allowed_tools=["tickets_get"], forbidden_tools=["admin_delete_record"]
    )
    assert "admin_delete_record" in [t.name for t in session.tools]

    text, is_error = session.call(
        "admin_delete_record", {"collection": "tickets", "record_id": "TKT-1"}
    )
    assert is_error and "Permission denied" in text
    assert session.blocked_calls[0].blocked_reason == "forbidden"
    # and crucially, the world is untouched
    assert world.maybe_find("tickets", "TKT-1") is not None


def test_unknown_tool_in_a_task_definition_fails_loudly():
    from agenteval.state import WorldError

    spec = TaskSpec(id="t", prompt="p", allowed_tools=["does.not.exist"])
    with pytest.raises(WorldError, match="unknown tool"):
        ToolSession(World({}), spec, Trajectory(task_id="t", agent="a"))


# -- budget ----------------------------------------------------------------- #


def test_step_budget_raises_once_exhausted():
    _, trajectory, session = make_session(max_steps=2)
    session.call("tickets_get", {"ticket_id": "TKT-1"})
    session.call("tickets_get", {"ticket_id": "TKT-1"})
    with pytest.raises(BudgetExceeded):
        session.call("tickets_get", {"ticket_id": "TKT-1"})
    assert trajectory.steps == 2


def test_blocked_calls_still_consume_budget():
    """Otherwise an agent could spin forever on rejected calls."""
    _, trajectory, session = make_session(max_steps=1)
    session.call("nope.nope", {})
    with pytest.raises(BudgetExceeded):
        session.call("tickets_get", {"ticket_id": "TKT-1"})


# -- unexpected handler failures -------------------------------------------- #


def test_an_unexpected_handler_exception_becomes_a_tool_error():
    """A bug in a tool handler must not take down the run. The agent gets an
    error result and can try something else; the harness records what happened.
    """
    from agenteval.registry import REGISTRY, ToolDef

    def boom(world):
        raise ZeroDivisionError("handler bug")

    REGISTRY["test_boom"] = ToolDef(
        name="test_boom",
        description="Explodes.",
        schema={"type": "object", "properties": {}, "required": [],
                "additionalProperties": False},
        handler=boom,
    )
    try:
        _, trajectory, session = make_session(allowed_tools=["test_boom"])
        text, is_error = session.call("test_boom", {})
        assert is_error
        assert "ZeroDivisionError: handler bug" in text
        assert trajectory.calls[0].blocked_reason is None  # it ran, it failed
    finally:
        del REGISTRY["test_boom"]


def test_p_int_produces_an_integer_schema():
    """Part of the P.* set tool authors pick from; unused today, so covered
    here rather than left to a future author to discover broken."""
    from agenteval import P

    spec = P.int("How many.", required=False)
    assert spec == {
        "schema": {"type": "integer", "description": "How many."},
        "required": False,
    }
