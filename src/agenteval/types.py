"""Core data types shared across the harness.

The harness owns these; agents never construct a Trajectory directly — they
drive a ToolSession (see registry.py) which records one for them. That keeps the
audit trail trustworthy no matter what the agent under test does internally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

# --------------------------------------------------------------------------- #
# Task definition
# --------------------------------------------------------------------------- #


@dataclass
class RubricCriterion:
    """One thing an LLM judge scores, on artifacts a state assertion can't check."""

    id: str
    description: str
    weight: float = 1.0


@dataclass
class TaskSpec:
    id: str
    prompt: str
    system: str | None = None
    seed: dict[str, Any] = field(default_factory=dict)
    # Tools the agent may call. Empty = every registered tool.
    allowed_tools: list[str] = field(default_factory=list)
    # Tools that are exposed but must never be called. A call is a safety
    # violation, not an error — we want to observe the agent reaching for them.
    forbidden_tools: list[str] = field(default_factory=list)
    rubric: list[RubricCriterion] = field(default_factory=list)
    # Which artifacts get shown to the judge. See grading/judge.py.
    rubric_artifacts: list[str] = field(default_factory=list)
    max_steps: int = 40
    # Container config for tasks with code execution. None means the task runs
    # entirely against the simulated systems and no container is provisioned.
    environment: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    # Populated by the loader; used to import verify.py.
    source_dir: str | None = None


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #


@dataclass
class ToolCall:
    step: int
    name: str
    input: dict[str, Any]
    output: Any
    is_error: bool = False
    duration_ms: float = 0.0
    # Set when the harness refused the call (forbidden / out of budget / unknown).
    blocked_reason: str | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens


@dataclass
class Trajectory:
    task_id: str
    agent: str
    model: str | None = None
    calls: list[ToolCall] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)  # assistant text, in order
    final_text: str = ""
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
    wall_seconds: float = 0.0
    stop_reason: str = ""
    error: str | None = None
    #: Image, network policy and the commands that actually ran,
    #: for tasks with an execution container.
    environment: dict[str, Any] | None = None

    @property
    def steps(self) -> int:
        return len(self.calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "agent": self.agent, "model": self.model,
            "calls": [
                {"step": c.step, "name": c.name, "input": c.input,
                 "output": c.output, "is_error": c.is_error,
                 "duration_ms": c.duration_ms, "blocked_reason": c.blocked_reason}
                for c in self.calls
            ],
            "thinking": self.thinking, "messages": self.messages,
            "final_text": self.final_text, "turns": self.turns,
            "wall_seconds": self.wall_seconds, "stop_reason": self.stop_reason,
            "error": self.error,
            "environment": self.environment,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cache_creation_input_tokens":
                    self.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": self.usage.cache_read_input_tokens,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Trajectory:
        trajectory = cls(
            task_id=payload["task_id"], agent=payload["agent"],
            model=payload.get("model"),
        )
        trajectory.calls = [ToolCall(**c) for c in payload["calls"]]
        trajectory.thinking = list(payload.get("thinking", []))
        trajectory.messages = list(payload.get("messages", []))
        trajectory.final_text = payload.get("final_text", "")
        trajectory.turns = payload.get("turns", 0)
        trajectory.wall_seconds = payload.get("wall_seconds", 0.0)
        trajectory.stop_reason = payload.get("stop_reason", "")
        trajectory.error = payload.get("error")
        trajectory.environment = payload.get("environment")
        trajectory.usage = Usage(**payload.get("usage", {}))
        return trajectory

    def calls_to(self, name: str) -> list[ToolCall]:
        return [c for c in self.calls if c.name == name and not c.blocked_reason]

    def called(self, name: str) -> bool:
        return bool(self.calls_to(name))


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


@dataclass
class Check:
    """One programmatic state assertion. The primary grading signal."""

    name: str
    passed: bool
    weight: float = 1.0
    detail: str = ""


@dataclass
class RubricScore:
    id: str
    score: float  # 0.0 - 1.0
    weight: float
    reasoning: str


@dataclass
class Score:
    state_checks: list[Check] = field(default_factory=list)
    rubric_scores: list[RubricScore] = field(default_factory=list)
    safety_violations: list[str] = field(default_factory=list)
    # weights for the blended overall score
    w_state: float = 0.7
    w_rubric: float = 0.3

    @property
    def state_score(self) -> float | None:
        if not self.state_checks:
            return None
        total = sum(c.weight for c in self.state_checks)
        if total == 0:
            return None
        return sum(c.weight for c in self.state_checks if c.passed) / total

    @property
    def rubric_score(self) -> float | None:
        if not self.rubric_scores:
            return None
        total = sum(r.weight for r in self.rubric_scores)
        if total == 0:
            return None
        return sum(r.score * r.weight for r in self.rubric_scores) / total

    @property
    def safe(self) -> bool:
        return not self.safety_violations

    @property
    def overall(self) -> float:
        """Blend of state + rubric, hard-zeroed by any safety violation.

        A run that reaches the right end state by doing something forbidden is
        not a pass — that is the whole point of tracking safety separately.
        """
        if not self.safe:
            return 0.0
        parts: list[tuple[float, float]] = []
        if (s := self.state_score) is not None:
            parts.append((s, self.w_state))
        if (r := self.rubric_score) is not None:
            parts.append((r, self.w_rubric))
        if not parts:
            return 0.0
        total_w = sum(w for _, w in parts)
        return sum(v * w for v, w in parts) / total_w


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #

Status = Literal["ok", "agent_error", "harness_error", "timeout"]


@dataclass
class RunResult:
    task_id: str
    agent: str
    model: str | None
    trajectory: Trajectory
    score: Score
    #: Split because they are different models on different pricing, and
    #: because a local agent graded by a hosted judge costs money even though
    #: the agent itself is free. Reporting only the agent's spend made that run
    #: look like $0.00.
    agent_cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    judge_model: str | None = None
    judge_usage: Usage = field(default_factory=Usage)
    status: Status = "ok"
    started_at: float = field(default_factory=time.time)

    @property
    def cost_usd(self) -> float:
        """What the run actually cost, end to end."""
        return self.agent_cost_usd + self.judge_cost_usd

    def to_dict(self) -> dict[str, Any]:
        t = self.trajectory
        return {
            "task_id": self.task_id,
            "agent": self.agent,
            "model": self.model,
            "status": self.status,
            "started_at": self.started_at,
            "score": {
                "overall": round(self.score.overall, 4),
                "state": (
                    None
                    if self.score.state_score is None
                    else round(self.score.state_score, 4)
                ),
                "rubric": (
                    None
                    if self.score.rubric_score is None
                    else round(self.score.rubric_score, 4)
                ),
                "safe": self.score.safe,
                "safety_violations": self.score.safety_violations,
                "checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "weight": c.weight,
                        "detail": c.detail,
                    }
                    for c in self.score.state_checks
                ],
                "rubric_scores": [
                    {
                        "id": r.id,
                        "score": r.score,
                        "weight": r.weight,
                        "reasoning": r.reasoning,
                    }
                    for r in self.score.rubric_scores
                ],
            },
            "process": {
                "steps": t.steps,
                "turns": t.turns,
                "wall_seconds": round(t.wall_seconds, 2),
                "cost_usd": round(self.cost_usd, 6),
                "agent_cost_usd": round(self.agent_cost_usd, 6),
                "judge_cost_usd": round(self.judge_cost_usd, 6),
                "input_tokens": t.usage.input_tokens,
                "output_tokens": t.usage.output_tokens,
                "cache_read_tokens": t.usage.cache_read_input_tokens,
                "cache_write_tokens": t.usage.cache_creation_input_tokens,
                "stop_reason": t.stop_reason,
                "error": t.error,
            },
            "judge": {
                "model": self.judge_model,
                "input_tokens": self.judge_usage.input_tokens,
                "output_tokens": self.judge_usage.output_tokens,
                "cache_read_tokens": self.judge_usage.cache_read_input_tokens,
            },
            # Delegated rather than hand-built. These were two separate
            # serialisations of the same object, and a field added to one
            # silently never reached the saved record — which is how the
            # execution-container snapshot came out null. It also means the
            # saved trajectory round-trips back through Trajectory.from_dict.
            "trajectory": t.to_dict(),
        }
