"""Agents under test, plus the spec-string factory used by the CLI."""

from __future__ import annotations

from typing import Any

from .base import Agent
from .claude import MODEL_ALIASES, ClaudeAgent, resolve_model
from .ollama import OllamaAgent
from .scripted import ScriptedAgent

__all__ = [
    "Agent",
    "ClaudeAgent",
    "OllamaAgent",
    "ScriptedAgent",
    "MODEL_ALIASES",
    "resolve_model",
    "build_agent",
]

#: Options only one backend understands. Passing `--max-turns` should not have
#: to know which agent it is talking to, so anything a backend cannot use is
#: dropped rather than raising a TypeError deep in construction.
_BACKEND_OPTIONS = {
    "claude": {"model", "effort", "max_turns", "max_tokens", "thinking",
               "show_thinking", "client"},
    "ollama": {"model", "host", "max_turns", "num_ctx", "temperature",
               "timeout", "client"},
}


def build_agent(spec: str, **overrides: Any) -> Agent:
    """Build an agent from a CLI spec string.

    Forms:
        claude                          -> ClaudeAgent on the default model
        claude:sonnet-5                 -> alias resolved against MODEL_ALIASES
        claude:claude-opus-4-8          -> full model id passed through
        claude:opus-5:medium            -> model plus effort level
        ollama                          -> OllamaAgent on the default model
        ollama:qwen2.5:7b-instruct      -> everything after the backend is the
                                           model tag
    """
    kind, _, rest = spec.partition(":")
    if kind not in _BACKEND_OPTIONS:
        raise ValueError(
            f"unknown agent {kind!r}. Built in: "
            f"{', '.join(sorted(_BACKEND_OPTIONS))}. "
            "Register your own by importing it and calling run_suite directly."
        )

    kwargs: dict[str, Any] = {}
    if kind == "claude":
        model, _, effort = rest.partition(":")
        if model:
            kwargs["model"] = model
        if effort:
            kwargs["effort"] = effort
    else:
        # Ollama model tags are themselves colon-separated (`qwen2.5:7b-instruct`),
        # so the whole remainder is the model — splitting again would silently
        # request `qwen2.5` and drop the variant.
        if rest:
            kwargs["model"] = rest

    accepted = _BACKEND_OPTIONS[kind]
    kwargs.update({k: v for k, v in overrides.items() if k in accepted})
    return ClaudeAgent(**kwargs) if kind == "claude" else OllamaAgent(**kwargs)
