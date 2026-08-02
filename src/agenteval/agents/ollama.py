"""Local models via Ollama.

Purpose here is not calibration — a 7B scoring 0.15 says nothing about where a
frontier model lands. Two other jobs:

* **Harness robustness.** ScriptedAgent replays a fixed list and never
  misbehaves. A small local model emits malformed arguments, invents tool
  names, loops, and blows the turn limit constantly. That exercises the error
  paths for real, for free.
* **A low anchor.** If the suite cannot separate a 7B from a frontier model,
  it is measuring nothing.

Deliberately kept separate from ClaudeAgent rather than sharing a base loop:
the message formats, tool encodings, and failure modes differ enough that the
shared abstraction would cost more clarity than it saves across two backends.

Uses Ollama's native `/api/chat` rather than its OpenAI-compatible endpoint,
because `options.num_ctx` is only settable on the native one — and leaving it
at the default is the single most destructive footgun here (see below).
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from ..registry import BudgetExceeded, ToolSession
from ..types import TaskSpec, Trajectory, Usage
from .claude import DEFAULT_SYSTEM

DEFAULT_HOST = "http://localhost:11434"

#: Ollama defaults to a small context and **silently discards** the oldest
#: tokens when a request exceeds it — no error, no warning. In a multi-turn
#: tool loop that means the task prompt quietly falls out of the window and the
#: model starts behaving inexplicably. Always set it explicitly.
DEFAULT_NUM_CTX = 32768

#: Reasoning distills (deepseek-r1 and friends) emit their scratchpad inline in
#: the content. Captured into the trajectory rather than left in the answer.
THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class OllamaError(Exception):
    pass


class OllamaAgent:
    """An agent backed by a local Ollama model."""

    #: No billable model. `cost_usd` treats None as zero rather than raising
    #: UnknownModel — local inference has no per-token price to report, and
    #: reporting a fabricated one would be worse than reporting nothing. The
    #: model identity travels in `name`, which is what reports key on.
    model = None

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        host: str = DEFAULT_HOST,
        max_turns: int = 20,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = 0.6,
        timeout: float = 600.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model_id = model
        self.host = host.rstrip("/")
        self.max_turns = max_turns
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        self._client = client
        self.name = f"ollama:{model}"

    # -- wire format -------------------------------------------------------- #

    def _tools(self, session: ToolSession) -> list[dict[str, Any]]:
        """Our JSON Schema is already what the `function` block wants."""
        return [
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.schema,
                },
            }
            for definition in session.tools
        ]

    @staticmethod
    def _arguments(raw: Any) -> dict[str, Any]:
        """Ollama returns parsed arguments; OpenAI-shaped clients send a JSON
        string. Accept either, and let a malformed one surface as a tool error
        rather than crashing the run — a small model producing invalid JSON is
        exactly the behaviour we are here to observe."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"__unparseable__": raw}
            return parsed if isinstance(parsed, dict) else {"__unparseable__": raw}
        return {}

    # -- request ------------------------------------------------------------ #

    async def _chat(
        self, client: httpx.AsyncClient, messages: list[dict[str, Any]],
        tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }
        try:
            response = await client.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self.host} — is `ollama serve` "
                f"running? ({exc})"
            ) from exc
        if response.status_code >= 400:
            raise OllamaError(
                f"Ollama returned {response.status_code}: {response.text[:400]}"
            )
        return response.json()

    # -- loop --------------------------------------------------------------- #

    async def run(
        self, task: TaskSpec, session: ToolSession, trajectory: Trajectory
    ) -> None:
        tools = self._tools(session)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": task.system or DEFAULT_SYSTEM},
            {"role": "user", "content": task.prompt},
        ]

        client = self._client or httpx.AsyncClient()
        owns_client = self._client is None
        try:
            for turn in range(1, self.max_turns + 1):
                trajectory.turns = turn
                data = await self._chat(client, messages, tools)
                message = data.get("message") or {}

                prompt_tokens = data.get("prompt_eval_count", 0) or 0
                trajectory.usage.add(
                    Usage(
                        input_tokens=prompt_tokens,
                        output_tokens=data.get("eval_count", 0) or 0,
                    )
                )
                self._check_context(prompt_tokens, trajectory)

                content = message.get("content") or ""
                if reasoning := THINK_BLOCK.findall(content):
                    trajectory.thinking.extend(r.strip() for r in reasoning)
                    content = THINK_BLOCK.sub("", content).strip()
                if content:
                    trajectory.messages.append(content)

                tool_calls = message.get("tool_calls") or []
                # Echo the assistant turn back verbatim so the model sees its
                # own tool calls alongside their results.
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        **({"tool_calls": tool_calls} if tool_calls else {}),
                    }
                )

                if not tool_calls:
                    trajectory.stop_reason = "end_turn"
                    break

                if self._dispatch(tool_calls, session, trajectory, messages):
                    break
            else:
                trajectory.error = f"turn limit of {self.max_turns} reached"
        finally:
            if owns_client:
                await client.aclose()

        trajectory.final_text = (
            trajectory.messages[-1] if trajectory.messages else ""
        )

    def _dispatch(
        self,
        tool_calls: list[dict[str, Any]],
        session: ToolSession,
        trajectory: Trajectory,
        messages: list[dict[str, Any]],
    ) -> bool:
        """Run each requested tool. Returns True when the run should stop."""
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name", "")
            arguments = self._arguments(function.get("arguments"))
            try:
                text, _ = session.call(name, arguments)
            except BudgetExceeded as exc:
                trajectory.error = str(exc)
                trajectory.stop_reason = "budget"
                return True
            # `tool_name` lets the model match a result to its call when it
            # issued several; older Ollama builds ignore the extra field.
            messages.append({"role": "tool", "tool_name": name, "content": text})
        return False

    def _check_context(self, prompt_tokens: int, trajectory: Trajectory) -> None:
        """Turn silent truncation into a visible finding.

        Past this point Ollama drops the oldest messages — including the task
        prompt — without saying so, and the resulting behaviour looks like a
        model failure rather than a configuration one.
        """
        if not prompt_tokens or prompt_tokens <= self.num_ctx * 0.9:
            return
        # Deduplicated on the stable prefix, not the whole message: the token
        # count climbs every turn, so matching the full string would append a
        # near-identical warning each time. Keyed off the trajectory rather
        # than the agent, which is shared across every run in the suite.
        marker = "context nearly full"
        if marker in (trajectory.error or ""):
            return
        warning = (
            f"{marker}: {prompt_tokens} tokens against num_ctx={self.num_ctx}; "
            "Ollama truncates silently past this, so raise num_ctx before "
            "trusting this run"
        )
        trajectory.error = (
            f"{trajectory.error + '; ' if trajectory.error else ''}{warning}"
        )
