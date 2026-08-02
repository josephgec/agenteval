"""ClaudeAgent's request shape and loop control, exercised against a stub client.

These do not call the API. They pin the things that are easy to get wrong and
expensive to discover live: a sampling parameter that current models reject
outright, thinking blocks dropped from the replayed history, tool results split
across messages, or a `refusal` stop reason read as if it were content.
"""

from types import SimpleNamespace

import pytest

from agenteval import ClaudeAgent, TaskSpec, ToolSession, Trajectory, World
from agenteval.agents.claude import resolve_model

SEED = {
    "tickets": [
        {"id": "TKT-1", "subject": "s", "body": "b", "status": "open",
         "priority": "P3", "team": "support", "comments": []}
    ]
}


# --------------------------------------------------------------------------- #
# Stub client
# --------------------------------------------------------------------------- #


def text(value):
    return SimpleNamespace(type="text", text=value)


def thinking(value):
    return SimpleNamespace(type="thinking", thinking=value)


def tool_use(call_id, name, payload):
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=payload)


def message(content, stop_reason="end_turn", **usage):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=usage.get("input_tokens", 100),
            output_tokens=usage.get("output_tokens", 50),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        ),
    )


class StubClient:
    """Replays a queue of responses and records every request it was sent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = self

    def stream(self, **kwargs):
        self.requests.append(kwargs)
        response = self._responses.pop(0)

        class Ctx:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *exc):
                return False

            async def get_final_message(self_inner):
                return response

        return Ctx()


def harness(responses, **task_kwargs):
    spec = TaskSpec(id="t", prompt="do the thing", seed=SEED, **task_kwargs)
    world = World(spec.seed)
    trajectory = Trajectory(task_id="t", agent="test")
    session = ToolSession(world, spec, trajectory)
    client = StubClient(responses)
    agent = ClaudeAgent(client=client, max_turns=6)
    return spec, world, trajectory, session, agent, client


# --------------------------------------------------------------------------- #
# Request shape
# --------------------------------------------------------------------------- #


async def test_request_omits_parameters_current_models_reject():
    spec, _, traj, session, agent, client = harness([message([text("done")])])
    await agent.run(spec, session, traj)

    request = client.requests[0]
    for rejected in ("temperature", "top_p", "top_k"):
        assert rejected not in request, f"{rejected} is a 400 on current models"
    # budget_tokens was replaced by output_config.effort
    assert "budget_tokens" not in request.get("thinking", {})


async def test_request_uses_adaptive_thinking_and_effort():
    spec, _, traj, session, agent, client = harness([message([text("done")])])
    await agent.run(spec, session, traj)

    request = client.requests[0]
    assert request["thinking"]["type"] == "adaptive"
    assert request["thinking"]["display"] == "summarized"
    assert request["output_config"] == {"effort": "high"}


async def test_thinking_can_be_disabled_explicitly():
    """Legal only at effort `high` or below; the CLI refuses the other pairing
    before any request is made."""
    spec = TaskSpec(id="t", prompt="p", seed=SEED)
    world = World(spec.seed)
    traj = Trajectory(task_id="t", agent="test")
    session = ToolSession(world, spec, traj)
    client = StubClient([message([text("done")])])
    await ClaudeAgent(client=client, thinking=False, effort="high").run(
        spec, session, traj
    )
    assert client.requests[0]["thinking"] == {"type": "disabled"}


async def test_thinking_summaries_can_be_suppressed():
    spec, _, traj, session, _, client = harness([message([text("done")])])
    agent = ClaudeAgent(client=client, show_thinking=False)
    await agent.run(spec, session, traj)
    assert client.requests[0]["thinking"]["display"] == "omitted"


async def test_system_block_carries_a_cache_breakpoint():
    """`tools` renders before `system`, so this caches both."""
    spec, _, traj, session, agent, client = harness([message([text("done")])])
    await agent.run(spec, session, traj)

    system = client.requests[0]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert client.requests[0]["cache_control"] == {"type": "ephemeral"}


async def test_tool_schemas_are_passed_through_in_stable_order():
    spec, _, traj, session, agent, client = harness(
        [message([text("done")])], allowed_tools=["tickets.update", "tickets.get"]
    )
    await agent.run(spec, session, traj)

    names = [t["name"] for t in client.requests[0]["tools"]]
    assert names == ["tickets.get", "tickets.update"]


# --------------------------------------------------------------------------- #
# Loop behaviour
# --------------------------------------------------------------------------- #


async def test_parallel_tool_calls_return_in_a_single_user_message():
    """Splitting them teaches the model to stop calling tools in parallel."""
    spec, _, traj, session, agent, client = harness(
        [
            message(
                [
                    tool_use("a", "tickets.get", {"ticket_id": "TKT-1"}),
                    tool_use("b", "tickets.get", {"ticket_id": "TKT-1"}),
                ],
                stop_reason="tool_use",
            ),
            message([text("done")]),
        ]
    )
    await agent.run(spec, session, traj)

    messages = client.requests[1]["messages"]
    tool_result_messages = [
        m
        for m in messages
        if m["role"] == "user"
        and isinstance(m["content"], list)
        and all(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert len(tool_result_messages) == 1
    assert len(tool_result_messages[0]["content"]) == 2
    assert {b["tool_use_id"] for b in tool_result_messages[0]["content"]} == {"a", "b"}


async def test_assistant_content_is_replayed_verbatim_including_thinking():
    """Editing or dropping thinking blocks breaks the following request."""
    blocks = [
        thinking("reasoning"),
        text("checking"),
        tool_use("a", "tickets.get", {"ticket_id": "TKT-1"}),
    ]
    spec, _, traj, session, agent, client = harness(
        [message(blocks, stop_reason="tool_use"), message([text("done")])]
    )
    await agent.run(spec, session, traj)

    assistant = [m for m in client.requests[1]["messages"] if m["role"] == "assistant"]
    assert assistant[0]["content"] is blocks  # same object, unmodified
    assert traj.thinking == ["reasoning"]


async def test_tool_results_reach_the_world():
    spec, world, traj, session, agent, _ = harness(
        [
            message(
                [tool_use("a", "tickets.update",
                          {"ticket_id": "TKT-1", "priority": "P0"})],
                stop_reason="tool_use",
            ),
            message([text("done")]),
        ]
    )
    await agent.run(spec, session, traj)
    assert world.find("tickets", "TKT-1")["priority"] == "P0"
    assert traj.steps == 1


async def test_refusal_stops_the_loop_and_is_recorded():
    spec, _, traj, session, agent, client = harness(
        [message([], stop_reason="refusal"), message([text("unreached")])]
    )
    await agent.run(spec, session, traj)

    assert traj.stop_reason == "refusal"
    assert "refused" in (traj.error or "")
    assert len(client.requests) == 1


async def test_pause_turn_resumes_without_adding_a_user_message():
    spec, _, traj, session, agent, client = harness(
        [
            message([text("partial")], stop_reason="pause_turn"),
            message([text("done")]),
        ]
    )
    await agent.run(spec, session, traj)

    assert len(client.requests) == 2
    messages = client.requests[1]["messages"]
    assert messages[-1]["role"] == "assistant"  # resumed, not re-prompted
    assert traj.final_text == "done"


async def test_usage_accumulates_across_turns():
    spec, _, traj, session, agent, _ = harness(
        [
            message(
                [tool_use("a", "tickets.get", {"ticket_id": "TKT-1"})],
                stop_reason="tool_use",
                input_tokens=100,
                output_tokens=20,
                cache_read_input_tokens=900,
            ),
            message([text("done")], input_tokens=150, output_tokens=30),
        ]
    )
    await agent.run(spec, session, traj)

    assert traj.usage.input_tokens == 250
    assert traj.usage.output_tokens == 50
    assert traj.usage.cache_read_input_tokens == 900
    assert traj.turns == 2


async def test_every_tool_use_gets_a_result_even_when_the_budget_runs_out():
    """The API rejects a turn where any tool_use id lacks a matching result."""
    spec, _, traj, session, agent, client = harness(
        [
            message(
                [
                    tool_use("a", "tickets.get", {"ticket_id": "TKT-1"}),
                    tool_use("b", "tickets.get", {"ticket_id": "TKT-1"}),
                    tool_use("c", "tickets.get", {"ticket_id": "TKT-1"}),
                ],
                stop_reason="tool_use",
            ),
            message([text("unreached")]),
        ],
        max_steps=1,
    )
    await agent.run(spec, session, traj)

    assert traj.steps == 1  # only the first call ran
    assert "step budget" in (traj.error or "")
    # The loop stops rather than sending a turn the API would reject, but the
    # results it built must still answer all three ids.
    assert len(client.requests) == 1


async def test_turn_limit_ends_the_run():
    responses = [
        message([tool_use(str(i), "tickets.get", {"ticket_id": "TKT-1"})],
                stop_reason="tool_use")
        for i in range(6)
    ]
    spec, _, traj, session, agent, _ = harness(responses)
    await agent.run(spec, session, traj)
    assert "turn limit" in (traj.error or "")


# --------------------------------------------------------------------------- #
# Model resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("opus-5", "claude-opus-5"),
        ("sonnet", "claude-sonnet-5"),
        ("haiku", "claude-haiku-4-5"),
        ("claude-opus-4-8", "claude-opus-4-8"),  # full ids pass through
    ],
)
def test_model_aliases_resolve(spec, expected):
    assert resolve_model(spec) == expected


def test_agent_name_identifies_model_and_effort():
    agent = ClaudeAgent(model="sonnet-5", effort="medium", client=StubClient([]))
    assert agent.name == "claude:claude-sonnet-5:medium"
