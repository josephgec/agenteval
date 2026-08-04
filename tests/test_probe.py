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
            tool = payload["tools"][0]["function"]["name"]
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
    assert result.called == {"short argument": True, "file as argument": False}


def test_both_shapes_are_probed(ollama):
    """One short argument and one whole-file argument, because they are
    different capabilities and this harness leans on the second."""
    ollama(lambda tool: True)
    assert set(probe_mod.run(["m"])[0].called) == {
        "short argument", "file as argument"
    }
    assert len(probe_mod.PROBES) == 2


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
