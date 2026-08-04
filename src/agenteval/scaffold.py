"""Telling a bad measurement apart from a bad model.

A run that ends having called no tools scores zero, and zero is a perfectly
ordinary thing for a weak model to score. But it is also what you get when the
model never had working tools at all — and those two are indistinguishable in
the results table, which is how a scaffold bug gets written up as a capability
finding.

This module exists because that happened here. `qwen2.5-coder:14b` answered a
HumanEval task with a correct `exec_write_file` call serialised as JSON *in its
reply text*, made no actual tool call, and scored 0.00. Nothing in the run said
anything was wrong: no exception, `stop_reason: end_turn`, a clean-looking
zero. The same model on an enterprise task called tools normally, so the fault
was in how the task was worded — a prompt that reads like a coding question
gets answered with code.

The checks below are heuristics and are reported as warnings, never as scores.
A warning does not move a number; it tells you the number may not mean what you
think. Getting that boundary wrong in the other direction — silently
"correcting" a run — would be much worse than a false positive.
"""

from __future__ import annotations

import json
import re

from .types import TaskSpec, Trajectory

#: A reply that is mostly a JSON object with these keys is a tool call the
#: model wrote out instead of making. Both spellings appear in the wild:
#: `arguments` is the OpenAI shape, `parameters` and `input` show up too.
_CALL_KEYS = ({"name", "arguments"}, {"name", "parameters"}, {"name", "input"})

_FENCE = re.compile(r"```(?:json|tool_code|python)?\s*(.*?)```", re.S)


def looks_like_an_unmade_tool_call(text: str) -> str | None:
    """The tool name, if this reply is a serialised call rather than prose."""
    for candidate in [text, *_FENCE.findall(text)]:
        candidate = candidate.strip()
        if not candidate.startswith("{"):
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            # Truncated or trailing prose. The opening shape is still telling,
            # and this is a warning rather than a verdict.
            head = candidate[:400]
            if '"name"' in head and ('"arguments"' in head or '"parameters"' in head):
                match = re.search(r'"name"\s*:\s*"([^"]+)"', head)
                return match.group(1) if match else "?"
            continue
        if isinstance(parsed, dict) and any(
            keys <= set(parsed) for keys in _CALL_KEYS
        ):
            return str(parsed.get("name", "?"))
    return None


def warnings(spec: TaskSpec, trajectory: Trajectory) -> list[str]:
    """Reasons to distrust this run's score, in plain words."""
    notes: list[str] = []
    if trajectory.calls:
        return notes

    # Only meaningful where calling something was the point. A task that can be
    # answered in prose is entitled to zero tool calls.
    if not spec.allowed_tools and not spec.environment:
        return notes

    name = looks_like_an_unmade_tool_call(trajectory.final_text)
    if name:
        notes.append(
            f"the model wrote a {name!r} call out as text instead of calling it, "
            "so this scored zero without the scaffold ever being exercised — "
            "the model or the prompt is not driving tool use, and the score is "
            "not a capability measurement"
        )
    elif not trajectory.final_text and not trajectory.messages:
        # The emptiest case, and the one an earlier version of this function
        # let through by requiring some text to complain about. Observed on 8
        # of 20 HumanEval runs: one turn, `end_turn`, no content, no calls,
        # status ok, scored 0.00. There is no reading of that as a capability
        # result — the model said nothing at all.
        notes.append(
            "the model returned an empty response and called nothing, so this "
            "run measured nothing at all — it is a failed request, not a score"
        )
    else:
        notes.append(
            "the run finished without calling a single tool, which for this "
            "task means nothing was measured"
        )
    return notes
