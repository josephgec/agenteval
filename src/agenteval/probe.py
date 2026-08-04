"""Does this model actually call tools?

A model that advertises tool support and then never emits a tool call scores
zero on every task here, and that zero is indistinguishable from a weak model
doing badly. `scaffold.py` catches it after the fact, per run. This catches it
*before* the run, in about a minute, for the price of two requests.

It exists because `qwen2.5-coder:14b` advertises `tools` in its Ollama
capabilities, and emits none — for any tool, on any prompt. Twenty HumanEval
instances and forty minutes were spent discovering that. The check below is the
experiment that finally isolated it, kept because the next model deserves to be
tested rather than trusted.

Two probes on purpose. One tool takes a small argument and one takes a large
string body, because "can call a search tool" and "can hand over a file's worth
of content in an argument" are different capabilities, and the second is the
one this harness leans on hardest.
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

PROBES = (
    ("short argument", SEARCH_TOOL, SEARCH_PROMPT, "tickets_search"),
    ("file as argument", WRITE_TOOL, WRITE_PROMPT, "exec_write_file"),
)


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
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
    return result


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
