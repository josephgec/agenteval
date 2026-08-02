"""Orchestration: build a world, run the agent in it, grade what it did.

Every run gets its own `World` built from the task seed, so runs never see each
other's state and can be executed concurrently or repeated for variance without
any reset logic.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from . import world as world_pkg  # noqa: F401  (import registers every tool)
from .agents.base import Agent
from .cost import cost_usd
from .grading.judge import JudgeError, LLMJudge
from .grading.safety import collect_safety_violations
from .registry import ToolSession
from .tasks import LoadedTask
from .types import RunResult, Score, Trajectory, Usage
from .state import World

ProgressFn = Callable[[str, RunResult | None], None]


@dataclass
class RunConfig:
    #: Repeats per task. >1 measures run-to-run variance, which for agentic
    #: tasks is often larger than the gap between two models.
    repeats: int = 1
    #: Concurrent runs in flight. Raise for throughput, lower on rate limits.
    concurrency: int = 4
    w_state: float = 0.7
    w_rubric: float = 0.3
    #: Off when a task has no rubric, or when --no-judge is passed.
    judge: LLMJudge | None = None


async def run_one(
    task: LoadedTask, agent: Agent, config: RunConfig, repeat: int = 0
) -> RunResult:
    world = World(task.spec.seed)
    trajectory = Trajectory(
        task_id=task.id, agent=agent.name, model=getattr(agent, "model", None)
    )
    session = ToolSession(world, task.spec, trajectory)

    status = "ok"
    started = time.time()
    clock = time.perf_counter()
    try:
        await agent.run(task.spec, session, trajectory)
    except Exception as exc:  # noqa: BLE001 - a crashed agent is a result, not a stop
        status = "agent_error"
        trajectory.error = f"{type(exc).__name__}: {exc}"
    trajectory.wall_seconds = time.perf_counter() - clock

    # Grade whatever state the agent left behind, even if it crashed — a run
    # that errors after doing the work is a different failure from one that
    # errors having done nothing, and the checks are what tell them apart.
    score = Score(w_state=config.w_state, w_rubric=config.w_rubric)
    try:
        score.state_checks = task.verify(world, trajectory)
    except Exception as exc:  # noqa: BLE001 - a broken verifier is a harness bug
        status = "harness_error"
        trajectory.error = (
            f"{trajectory.error + '; ' if trajectory.error else ''}"
            f"verifier raised {type(exc).__name__}: {exc}"
        )

    score.safety_violations = collect_safety_violations(world, trajectory, session)
    if task.safety:
        score.safety_violations.extend(task.safety(world, trajectory))

    judge_usage = Usage()
    judge_model = getattr(config.judge, "model", None) if config.judge else None
    if config.judge and task.spec.rubric:
        try:
            outcome = await config.judge.score(task.spec, world, trajectory)
            score.rubric_scores = outcome.scores
            judge_usage = outcome.usage
        except JudgeError as exc:
            # Left explicit rather than silently degrading to state-only
            # scoring, which would make the run look better than it was. The
            # failed attempt was still billed, so its usage is kept.
            status = "harness_error"
            judge_usage = exc.usage
            trajectory.error = (
                f"{trajectory.error + '; ' if trajectory.error else ''}{exc}"
            )

    agent_model = getattr(agent, "model", None)
    return RunResult(
        task_id=task.id,
        agent=agent.name,
        model=agent_model,
        trajectory=trajectory,
        score=score,
        agent_cost_usd=cost_usd(agent_model, trajectory.usage),
        # Priced against the judge's own model: it is frequently a different
        # one from the agent's, and with a local agent it is the only spend.
        judge_cost_usd=cost_usd(judge_model, judge_usage),
        judge_model=judge_model,
        judge_usage=judge_usage,
        status=status,  # type: ignore[arg-type]
        started_at=started,
    )


async def run_suite(
    tasks: list[LoadedTask],
    agent: Agent,
    config: RunConfig | None = None,
    on_progress: ProgressFn | None = None,
) -> list[RunResult]:
    config = config or RunConfig()
    semaphore = asyncio.Semaphore(config.concurrency)

    async def guarded(task: LoadedTask, repeat: int) -> RunResult:
        async with semaphore:
            if on_progress:
                on_progress(task.id, None)
            result = await run_one(task, agent, config, repeat)
            if on_progress:
                on_progress(task.id, result)
            return result

    jobs: list[Awaitable[RunResult]] = [
        guarded(task, r)
        for task in tasks
        for r in range(config.repeats)
    ]
    return list(await asyncio.gather(*jobs))
