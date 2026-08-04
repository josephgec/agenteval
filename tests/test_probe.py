"""Checking a model emits tool calls before spending a run on it.

The probe talks to Ollama, so everything here stubs the transport. What is
being tested is the verdict: which answers count as usable, and which are the
quiet failure this exists to catch.
"""

import httpx
import pytest

from agenteval import cli, probe as probe_mod


def _transport(handler):
    """An httpx client whose responses come from `handler(payload)`."""
    def respond(request: httpx.Request) -> httpx.Response:
        import json

        return httpx.Response(200, json=handler(json.loads(request.content)))

    return httpx.MockTransport(respond)


@pytest.fixture
def ollama(monkeypatch):
    """Install a fake Ollama whose behaviour a test chooses per tool."""
    def install(behaviour):
        def handler(payload):
            # The continuation probe sends two tools and, on its second turn,
            # a transcript that already contains a tool result. Answering with
            # tools[0] there would return a write call where the probe is
            # looking for exec_bash, and every model would fail it.
            answered = any(m["role"] == "tool" for m in payload["messages"])
            tool = ("exec_bash" if answered
                    else payload["tools"][0]["function"]["name"])
            if behaviour(tool):
                return {"message": {"role": "assistant", "content": "",
                                    "tool_calls": [{"function": {
                                        "name": tool, "arguments": {}}}]}}
            return {"message": {"role": "assistant",
                                "content": '{"name": "%s", "arguments": {}}' % tool}}

        real = httpx.AsyncClient

        def build(*args, **kwargs):
            return real(transport=_transport(handler), **{
                k: v for k, v in kwargs.items() if k == "timeout"
            })

        monkeypatch.setattr(probe_mod.httpx, "AsyncClient", build)

    return install


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #


def test_a_model_that_calls_both_tools_is_usable(ollama):
    ollama(lambda tool: True)
    result = probe_mod.run(["good-model"])[0]
    assert result.usable
    assert result.summary() == "calls tools"


def test_a_model_that_calls_nothing_is_not(ollama):
    """The observed failure: qwen2.5-coder:14b advertises tool support and
    emits none, for any tool, on any prompt."""
    ollama(lambda tool: False)
    result = probe_mod.run(["talks-instead"])[0]
    assert not result.usable
    assert "answered in text" in result.summary()


def test_calling_only_the_easy_tool_is_still_a_failure(ollama):
    """A model that manages a short argument but not a file body passes the
    enterprise tasks and fails every code benchmark — worse than failing
    outright, because it looks like a capability finding."""
    ollama(lambda tool: tool == "tickets_search")
    result = probe_mod.run(["half-works"])[0]
    assert not result.usable
    assert "exec_write_file" in result.summary()
    assert result.called["short argument"] is True
    assert result.called["file as argument"] is False


def test_every_shape_that_comes_apart_is_probed(ollama):
    """Three capabilities that are genuinely separate: a short argument, a
    file's worth of content as an argument, and continuing to drive the loop
    once results start coming back. Each was added because the previous set
    passed a model that then failed."""
    ollama(lambda tool: True)
    assert set(probe_mod.run(["m"])[0].called) == {
        "short argument", "file as argument", probe_mod.CONTINUES
    }


def test_the_text_reply_is_kept_for_reading(ollama):
    """So the failure can be recognised rather than just counted."""
    ollama(lambda tool: False)
    result = probe_mod.run(["talks-instead"])[0]
    assert "tickets_search" in result.replies["short argument"]


def test_an_unreachable_ollama_is_reported_not_raised(monkeypatch):
    def refuse(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(probe_mod.httpx, "AsyncClient", refuse)
    result = probe_mod.run(["nothing-there"])[0]
    assert not result.usable
    assert "could not be reached" in result.summary()


def test_probing_is_deterministic(ollama):
    """A capability check, so a model that calls tools only sometimes must not
    pass by luck."""
    seen = {}

    def handler(payload):
        seen["temperature"] = payload["options"]["temperature"]
        return {"message": {"content": "", "tool_calls": [
            {"function": {"name": payload["tools"][0]["function"]["name"]}}]}}

    real = httpx.AsyncClient
    import agenteval.probe as p

    def build(*args, **kwargs):
        return real(transport=_transport(handler))

    p.httpx.AsyncClient = build
    try:
        probe_mod.run(["m"])
    finally:
        p.httpx.AsyncClient = real
    assert seen["temperature"] == 0.0


def test_models_are_probed_one_at_a_time(ollama):
    """Local inference is memory-bound; two models loaded at once measures the
    swapping rather than the models."""
    ollama(lambda tool: True)
    results = probe_mod.run(["a", "b"])
    assert [r.model for r in results] == ["a", "b"]


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #


def test_probe_reports_each_model(ollama, capsys, monkeypatch):
    ollama(lambda tool: tool == "tickets_search")
    monkeypatch.setenv("COLUMNS", "200")
    code = cli.main(["probe", "--model", "half-works"])
    out = capsys.readouterr().out
    assert "half-works" in out
    assert "answered in text" in out
    assert code == 1  # nothing usable


def test_probe_exits_zero_when_something_works(ollama, monkeypatch):
    ollama(lambda tool: True)
    monkeypatch.setenv("COLUMNS", "200")
    assert cli.main(["probe", "--model", "good"]) == 0


def test_probe_defaults_to_everything_ollama_has(ollama, monkeypatch):
    ollama(lambda tool: True)
    monkeypatch.setattr(probe_mod, "available_models", lambda host: ["one", "two"])
    monkeypatch.setenv("COLUMNS", "200")
    assert cli.main(["probe"]) == 0


def test_probe_says_so_when_ollama_has_nothing(monkeypatch, capsys):
    monkeypatch.setattr(probe_mod, "available_models", lambda host: [])
    assert cli.main(["probe"]) == 2
    assert "Is Ollama running" in capsys.readouterr().out


def test_listing_models_survives_ollama_being_down(monkeypatch):
    def refuse(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(probe_mod.httpx, "get", refuse)
    assert probe_mod.available_models() == []


# --------------------------------------------------------------------------- #
# Keeping going after a result
# --------------------------------------------------------------------------- #
#
# Added after lfm2.5:8b passed both single-turn probes and then scored zero
# across twenty benchmark instances, half of them without emitting a tool call
# at all. One turn of evidence was not enough.


def _conversation(handler):
    """A fake Ollama that answers based on how many turns have happened."""
    import json as _json

    turns = {"n": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        payload = _json.loads(request.content)
        # The continuation probe is the only one that sends two tools.
        multiturn = len(payload["tools"]) == 2
        if multiturn:
            turns["n"] += 1
        return httpx.Response(200, json=handler(payload, turns["n"], multiturn))

    return httpx.MockTransport(respond)


def _install(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(
        probe_mod.httpx, "AsyncClient",
        lambda *a, **k: real(transport=_conversation(handler)),
    )


def _tool_call(name):
    return {"message": {"role": "assistant", "content": "",
                        "tool_calls": [{"function": {"name": name,
                                                     "arguments": {}}}]}}


def _text(body):
    return {"message": {"role": "assistant", "content": body}}


def test_a_model_that_stops_driving_the_loop_is_caught(monkeypatch):
    """The lfm2.5 failure exactly: a clean first call, then narration."""
    def handler(payload, turn, multiturn):
        if multiturn and turn == 2:
            return _text("The file has been written. Next you should run it.")
        return _tool_call(payload["tools"][0]["function"]["name"])

    _install(monkeypatch, handler)
    result = probe_mod.run(["stops-after-one"])[0]
    assert result.called[probe_mod.CONTINUES] is False
    assert not result.usable
    assert "a second tool after a result" in result.summary()


def test_a_model_that_keeps_going_passes(monkeypatch):
    def handler(payload, turn, multiturn):
        if multiturn and turn == 2:
            return _tool_call("exec_bash")
        return _tool_call(payload["tools"][0]["function"]["name"])

    _install(monkeypatch, handler)
    result = probe_mod.run(["keeps-going"])[0]
    assert result.called[probe_mod.CONTINUES] is True
    assert result.usable


def test_a_model_that_never_starts_fails_the_continuation_too(monkeypatch):
    """No first call means there is nothing to continue from, and that is a
    failure rather than a skipped probe."""
    _install(monkeypatch, lambda payload, turn, multiturn: _text("{...}"))
    result = probe_mod.run(["never-calls"])[0]
    assert result.called[probe_mod.CONTINUES] is False


def test_the_narration_is_kept_for_reading(monkeypatch):
    def handler(payload, turn, multiturn):
        if multiturn and turn == 2:
            return _text("You should now run the file yourself.")
        return _tool_call(payload["tools"][0]["function"]["name"])

    _install(monkeypatch, handler)
    result = probe_mod.run(["stops"])[0]
    assert "run the file yourself" in result.replies[probe_mod.CONTINUES]


def test_the_second_turn_replays_the_tool_result(monkeypatch):
    """The probe has to answer the first call the way the real loop does, or it
    is testing a conversation the agent never has."""
    seen = {}

    def handler(payload, turn, multiturn):
        if multiturn and turn == 2:
            seen["roles"] = [m["role"] for m in payload["messages"]]
            return _tool_call("exec_bash")
        return _tool_call(payload["tools"][0]["function"]["name"])

    _install(monkeypatch, handler)
    probe_mod.run(["m"])
    assert seen["roles"] == ["user", "assistant", "tool", "user"]
