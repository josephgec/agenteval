"""Does this model actually call tools?

A model that advertises tool support and then never emits a tool call scores
zero on every task here, and that zero is indistinguishable from a weak model
doing badly. `scaffold.py` catches it after the fact, per run. This catches it
*before* the run, in about a minute, for the price of four requests.

It exists because `qwen2.5-coder:14b` advertises `tools` in its Ollama
capabilities, and emits none — for any tool, on any prompt. Twenty HumanEval
instances and forty minutes were spent discovering that. The checks below are
the experiments that isolated it, kept because the next model deserves to be
tested rather than trusted.

Three probes, and each one exists because the previous set was not enough.

*A short argument.* The easy case: one small string.

*A file as an argument.* Handing over a file's worth of content is a different
capability from calling a search tool, and it is the one this harness leans on
hardest. Models that manage the first and not the second sail through the
enterprise tasks and fail every code benchmark.

*Keeping going after a result.* Added after `lfm2.5:8b` passed both of the
above and then scored zero across twenty consecutive benchmark instances, half
of them without emitting a single tool call. A model can make one clean call
and then stop driving the loop the moment tool results start coming back, and
no amount of single-turn evidence can see that. This probe answers the first
call the way the real loop does and asks for a dependent second one.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx

from .agents.ollama import DEFAULT_HOST, DEFAULT_NUM_CTX

#: A trivial call with one short argument — the easy case.
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "tickets_search",
        "description": "Search the ticket system.",
        "parameters": {
            "type": "object",
            "properties": {"status": {"type": "string",
                                      "description": "open, closed"}},
            "required": ["status"],
        },
    },
}
SEARCH_PROMPT = "Find all the open tickets."

#: A call whose argument is a whole file. Every code benchmark here depends on
#: this shape, and it is the one models most often answer in prose instead.
WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "exec_write_file",
        "description": "Write a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write."},
                "content": {"type": "string", "description": "Full contents."},
            },
            "required": ["path", "content"],
        },
    },
}
WRITE_PROMPT = (
    "Use your tools to create /workspace/solution.py containing a Python "
    "function `square(n)` that returns n squared."
)

#: The second turn of the multi-turn probe: having been told the file was
#: written, does the model keep driving tools, or does it start narrating?
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "exec_bash",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string",
                                       "description": "The command to run."}},
            "required": ["command"],
        },
    },
}
CONTINUE_PROMPT = "Now run that file with python and tell me what it prints."

PROBES = (
    ("short argument", SEARCH_TOOL, SEARCH_PROMPT, "tickets_search"),
    ("file as argument", WRITE_TOOL, WRITE_PROMPT, "exec_write_file"),
)

#: Reported alongside the single-turn probes but measured separately, because
#: it is a different capability and the two come apart.
CONTINUES = "keeps going after a result"


@dataclass
class ProbeResult:
    model: str
    called: dict[str, bool] = field(default_factory=dict)
    replies: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def usable(self) -> bool:
        """Every probe has to pass.

        Not a majority: a model that manages the short argument and not the
        file body will fail every code benchmark here while looking fine on the
        enterprise tasks, which is a worse outcome than failing outright.
        """
        return bool(self.called) and all(self.called.values()) and not self.error

    def summary(self) -> str:
        if self.error:
            return f"could not be reached: {self.error}"
        if self.usable:
            return "calls tools"
        # Named by the tool rather than the probe label: "exec_write_file" is
        # something you can go and look at, "file as argument" is not.
        tools = {label: expected for label, _, _, expected in PROBES}
        tools[CONTINUES] = "a second tool after a result"
        failed = [tools.get(label, label) for label, ok in self.called.items()
                  if not ok]
        return f"answered in text instead of calling: {', '.join(failed)}"


async def probe_one(
    model: str,
    host: str = DEFAULT_HOST,
    num_ctx: int = DEFAULT_NUM_CTX,
    timeout: float = 300.0,
) -> ProbeResult:
    result = ProbeResult(model=model)
    # The whole exchange is guarded, not just the request. Opening the client
    # can fail too, and a probe that raises instead of reporting is no better
    # than the failure it is meant to catch.
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for label, tool, prompt, expected in PROBES:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "tools": [tool],
                    "stream": False,
                    # Deterministic: this is a capability check, and a model
                    # that calls tools only sometimes should not pass by luck.
                    "options": {"num_ctx": num_ctx, "temperature": 0.0},
                }
                reply = await client.post(
                    f"{host.rstrip('/')}/api/chat", json=payload
                )
                reply.raise_for_status()
                message = reply.json().get("message", {})
                calls = message.get("tool_calls") or []
                names = [c.get("function", {}).get("name") for c in calls]
                result.called[label] = expected in names
                result.replies[label] = (message.get("content") or "")[:400]

            # The multi-turn probe. A model can make a clean first call and
            # then stop driving the loop once tool results start coming back,
            # and a single-turn check cannot see the difference — `lfm2.5:8b`
            # passed both probes above and then scored zero on twenty
            # consecutive benchmark instances, half of them without emitting a
            # single tool call. One turn of evidence was not enough.
            await _probe_continuation(client, model, host, num_ctx, result)
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
    return result


async def _probe_continuation(
    client: httpx.AsyncClient, model: str, host: str, num_ctx: int,
    result: ProbeResult,
) -> None:
    """Ask for a call, answer it, and see whether a second call follows."""
    tools = [WRITE_TOOL, BASH_TOOL]
    messages: list[dict[str, Any]] = [{"role": "user", "content": WRITE_PROMPT}]

    async def turn() -> dict[str, Any]:
        reply = await client.post(f"{host.rstrip('/')}/api/chat", json={
            "model": model, "messages": messages, "tools": tools,
            "stream": False,
            "options": {"num_ctx": num_ctx, "temperature": 0.0},
        })
        reply.raise_for_status()
        return reply.json().get("message", {})

    first = await turn()
    calls = first.get("tool_calls") or []
    if not calls:
        result.called[CONTINUES] = False
        result.replies[CONTINUES] = (first.get("content") or "")[:400]
        return

    # Answer the call the way the real loop does, then ask for the next step.
    messages.append({"role": "assistant", "content": first.get("content") or "",
                     "tool_calls": calls})
    messages.append({"role": "tool", "name": "exec_write_file",
                     "content": "Wrote 46 characters to /workspace/solution.py."})
    messages.append({"role": "user", "content": CONTINUE_PROMPT})

    second = await turn()
    names = [c.get("function", {}).get("name")
             for c in (second.get("tool_calls") or [])]
    result.called[CONTINUES] = "exec_bash" in names
    result.replies[CONTINUES] = (second.get("content") or "")[:400]


async def probe(
    models: list[str], host: str = DEFAULT_HOST, num_ctx: int = DEFAULT_NUM_CTX
) -> list[ProbeResult]:
    # Sequentially: local inference is memory-bound, and two models loaded at
    # once on one machine measures the swapping rather than the models.
    return [await probe_one(m, host=host, num_ctx=num_ctx) for m in models]


def available_models(host: str = DEFAULT_HOST, timeout: float = 20.0) -> list[str]:
    try:
        reply = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        reply.raise_for_status()
        return [m["name"] for m in reply.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def run(models: list[str], host: str = DEFAULT_HOST) -> list[ProbeResult]:
    return asyncio.run(probe(models, host=host))
