"""OllamaAgent: wire format, loop control, and the failure modes local models
actually produce.

Driven by httpx.MockTransport, so these run offline with no Ollama server. The
emphasis is on malformed output — invalid JSON arguments, invented tool names,
runaway loops — because tolerating those is the entire reason this backend
exists.
"""

import json

import httpx
import pytest

from agenteval import OllamaAgent, TaskSpec, ToolSession, Trajectory, World, cost_usd
from agenteval.agents.ollama import DEFAULT_NUM_CTX, OllamaError

SEED = {
    "tickets": [
        {"id": "TKT-1", "subject": "s", "body": "b", "status": "open",
         "priority": "P3", "team": "support", "comments": []}
    ]
}


def reply(content="", tool_calls=None, prompt_tokens=100, eval_tokens=20):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "message": message,
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": eval_tokens,
    }


def call(name, arguments):
    return {"function": {"name": name, "arguments": arguments}}


def harness(responses, status=200, exc=None, **agent_kwargs):
    """Wire an agent to a scripted sequence of Ollama replies."""
    spec = TaskSpec(id="t", prompt="do the thing", seed=SEED,
                    **agent_kwargs.pop("task_kwargs", {}))
    world = World(spec.seed)
    trajectory = Trajectory(task_id="t", agent="test")
    session = ToolSession(world, spec, trajectory)

    sent: list[dict] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if exc:
            raise exc
        sent.append(json.loads(request.content))
        if status >= 400:
            return httpx.Response(status, text="boom")
        return httpx.Response(200, json=queue.pop(0))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    agent = OllamaAgent(client=client, max_turns=6, **agent_kwargs)
    return spec, world, trajectory, session, agent, sent


# --------------------------------------------------------------------------- #
# Request shape
# --------------------------------------------------------------------------- #


async def test_num_ctx_is_always_set_explicitly():
    """Ollama's default context is small and it truncates silently — leaving
    this unset drops the task prompt mid-run with no error."""
    spec, _, traj, session, agent, sent = harness([reply("done")])
    await agent.run(spec, session, traj)
    assert sent[0]["options"]["num_ctx"] == DEFAULT_NUM_CTX


async def test_num_ctx_is_configurable():
    spec, _, traj, session, agent, sent = harness([reply("done")], num_ctx=8192)
    await agent.run(spec, session, traj)
    assert sent[0]["options"]["num_ctx"] == 8192


async def test_tools_are_sent_in_openai_function_format():
    spec, _, traj, session, agent, sent = harness(
        [reply("done")], task_kwargs={"allowed_tools": ["tickets.get"]}
    )
    await agent.run(spec, session, traj)

    [tool] = sent[0]["tools"]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "tickets.get"
    assert tool["function"]["description"]
    # Our JSON Schema is passed through unchanged as `parameters`.
    assert tool["function"]["parameters"]["required"] == ["ticket_id"]


async def test_request_is_non_streaming_and_names_the_model():
    spec, _, traj, session, agent, sent = harness(
        [reply("done")], model="qwen2.5:7b-instruct"
    )
    await agent.run(spec, session, traj)
    assert sent[0]["stream"] is False
    assert sent[0]["model"] == "qwen2.5:7b-instruct"


async def test_system_prompt_and_task_prompt_open_the_conversation():
    spec, _, traj, session, agent, sent = harness([reply("done")])
    await agent.run(spec, session, traj)
    messages = sent[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "do the thing"}


# --------------------------------------------------------------------------- #
# Tool loop
# --------------------------------------------------------------------------- #


async def test_tool_calls_reach_the_world_and_results_go_back():
    spec, world, traj, session, agent, sent = harness(
        [
            reply(tool_calls=[
                call("tickets.update", {"ticket_id": "TKT-1", "priority": "P0"})
            ]),
            reply("all done"),
        ]
    )
    await agent.run(spec, session, traj)

    assert world.find("tickets", "TKT-1")["priority"] == "P0"
    tool_messages = [m for m in sent[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_name"] == "tickets.update"
    assert traj.final_text == "all done"


async def test_parallel_tool_calls_are_all_executed():
    spec, _, traj, session, agent, sent = harness(
        [
            reply(tool_calls=[
                call("tickets.get", {"ticket_id": "TKT-1"}),
                call("tickets.get", {"ticket_id": "TKT-1"}),
            ]),
            reply("done"),
        ]
    )
    await agent.run(spec, session, traj)
    assert traj.steps == 2
    assert len([m for m in sent[1]["messages"] if m["role"] == "tool"]) == 2


async def test_the_assistant_turn_is_echoed_back_with_its_tool_calls():
    calls = [call("tickets.get", {"ticket_id": "TKT-1"})]
    spec, _, traj, session, agent, sent = harness(
        [reply("checking", tool_calls=calls), reply("done")]
    )
    await agent.run(spec, session, traj)

    assistant = [m for m in sent[1]["messages"] if m["role"] == "assistant"]
    assert assistant[0]["content"] == "checking"
    assert assistant[0]["tool_calls"] == calls


async def test_usage_maps_from_ollamas_counters():
    spec, _, traj, session, agent, _ = harness(
        [
            reply(tool_calls=[call("tickets.get", {"ticket_id": "TKT-1"})],
                  prompt_tokens=500, eval_tokens=40),
            reply("done", prompt_tokens=700, eval_tokens=60),
        ]
    )
    await agent.run(spec, session, traj)
    assert traj.usage.input_tokens == 1200
    assert traj.usage.output_tokens == 100
    assert traj.turns == 2


async def test_local_inference_reports_no_dollar_cost():
    """`model` is None so cost_usd returns zero rather than raising for an
    unpriced model. The model identity travels in `name`."""
    agent = OllamaAgent(model="qwen2.5:7b-instruct")
    assert agent.model is None
    assert agent.name == "ollama:qwen2.5:7b-instruct"
    assert cost_usd(agent.model, Trajectory("t", "a").usage) == 0.0


# --------------------------------------------------------------------------- #
# Malformed output — the reason this backend exists
# --------------------------------------------------------------------------- #


async def test_arguments_may_arrive_as_a_json_string():
    spec, world, traj, session, agent, _ = harness(
        [
            reply(tool_calls=[
                call("tickets.update",
                     json.dumps({"ticket_id": "TKT-1", "priority": "P1"}))
            ]),
            reply("done"),
        ]
    )
    await agent.run(spec, session, traj)
    assert world.find("tickets", "TKT-1")["priority"] == "P1"


async def test_unparseable_arguments_become_a_tool_error_not_a_crash():
    spec, _, traj, session, agent, _ = harness(
        [
            reply(tool_calls=[call("tickets.get", "{not json at all")]),
            reply("sorry"),
        ]
    )
    await agent.run(spec, session, traj)

    [recorded] = traj.calls
    assert recorded.blocked_reason == "bad_args"
    assert "__unparseable__" in recorded.input
    assert traj.final_text == "sorry"  # the loop continued


async def test_a_json_scalar_instead_of_an_object_is_handled():
    spec, _, traj, session, agent, _ = harness(
        [reply(tool_calls=[call("tickets.get", "42")]), reply("done")]
    )
    await agent.run(spec, session, traj)
    assert traj.calls[0].blocked_reason == "bad_args"


async def test_an_invented_tool_name_is_reported_back_to_the_model():
    spec, _, traj, session, agent, sent = harness(
        [reply(tool_calls=[call("tickets.magic", {})]), reply("ok")]
    )
    await agent.run(spec, session, traj)

    assert traj.calls[0].blocked_reason == "unknown"
    tool_message = [m for m in sent[1]["messages"] if m["role"] == "tool"][0]
    assert "Unknown tool" in tool_message["content"]


async def test_a_runaway_loop_is_stopped_by_the_turn_limit():
    spec, _, traj, session, agent, _ = harness(
        [reply(tool_calls=[call("tickets.get", {"ticket_id": "TKT-1"})])] * 6
    )
    await agent.run(spec, session, traj)
    assert "turn limit" in traj.error


async def test_the_step_budget_ends_the_run():
    spec, _, traj, session, agent, _ = harness(
        [reply(tool_calls=[call("tickets.get", {"ticket_id": "TKT-1"})])] * 6,
        task_kwargs={"max_steps": 2},
    )
    await agent.run(spec, session, traj)
    assert traj.steps == 2
    assert traj.stop_reason == "budget"
    assert "step budget" in traj.error


# --------------------------------------------------------------------------- #
# Reasoning models and operational failures
# --------------------------------------------------------------------------- #


async def test_think_blocks_are_captured_not_left_in_the_answer():
    """deepseek-r1 and similar distills emit their scratchpad inline."""
    spec, _, traj, session, agent, _ = harness(
        [reply("<think>let me consider this</think>The answer is 4.")]
    )
    await agent.run(spec, session, traj)
    assert traj.thinking == ["let me consider this"]
    assert traj.final_text == "The answer is 4."


async def test_multiple_think_blocks_are_all_captured():
    spec, _, traj, session, agent, _ = harness(
        [reply("<think>one</think>mid<think>two</think>end")]
    )
    await agent.run(spec, session, traj)
    assert traj.thinking == ["one", "two"]


async def test_nearing_the_context_limit_is_recorded_as_a_finding():
    """Silent truncation looks like a model failure; this makes it visible."""
    spec, _, traj, session, agent, _ = harness(
        [reply("done", prompt_tokens=7900)], num_ctx=8192
    )
    await agent.run(spec, session, traj)
    assert "context nearly full" in traj.error
    assert "num_ctx=8192" in traj.error


async def test_the_context_warning_is_not_repeated_every_turn():
    spec, _, traj, session, agent, _ = harness(
        [
            reply(tool_calls=[call("tickets.get", {"ticket_id": "TKT-1"})],
                  prompt_tokens=7900),
            reply("done", prompt_tokens=7950),
        ],
        num_ctx=8192,
    )
    await agent.run(spec, session, traj)
    assert traj.error.count("context nearly full") == 1


async def test_comfortable_context_usage_is_not_flagged():
    spec, _, traj, session, agent, _ = harness(
        [reply("done", prompt_tokens=1000)], num_ctx=8192
    )
    await agent.run(spec, session, traj)
    assert traj.error is None


async def test_a_dead_server_explains_itself():
    spec, _, traj, session, agent, _ = harness(
        [], exc=httpx.ConnectError("connection refused")
    )
    with pytest.raises(OllamaError, match="is `ollama serve` running"):
        await agent.run(spec, session, traj)


async def test_an_http_error_surfaces_the_body():
    spec, _, traj, session, agent, _ = harness([reply()], status=500)
    with pytest.raises(OllamaError, match="returned 500"):
        await agent.run(spec, session, traj)


async def test_an_empty_response_does_not_crash():
    """Some models return no content and no tool calls."""
    spec, _, traj, session, agent, _ = harness([{"message": {}, "done": True}])
    await agent.run(spec, session, traj)
    assert traj.final_text == ""
    assert traj.stop_reason == "end_turn"


# --------------------------------------------------------------------------- #
# Spec parsing
# --------------------------------------------------------------------------- #


def test_the_model_tag_survives_its_own_colons():
    """`ollama:qwen2.5:7b-instruct` must not be split into model `qwen2.5`."""
    from agenteval import build_agent

    agent = build_agent("ollama:qwen2.5:7b-instruct")
    assert isinstance(agent, OllamaAgent)
    assert agent.model_id == "qwen2.5:7b-instruct"
    assert agent.name == "ollama:qwen2.5:7b-instruct"


def test_a_bare_ollama_spec_uses_the_default_model():
    from agenteval import build_agent

    assert build_agent("ollama").model_id == "qwen2.5:7b-instruct"


def test_backend_specific_options_are_filtered_not_fatal():
    """The CLI passes every flag to whichever backend was chosen."""
    from agenteval import build_agent

    ollama = build_agent("ollama:qwen2.5:7b-instruct", max_turns=3,
                         thinking=False, num_ctx=4096)
    assert ollama.max_turns == 3 and ollama.num_ctx == 4096

    claude = build_agent("claude:opus-5", max_turns=3, num_ctx=4096,
                         host="http://x")
    assert claude.max_turns == 3
