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

from . import exec as exec_pkg  # noqa: F401  (registers the exec tools)
from . import world as world_pkg  # noqa: F401  (import registers every tool)
from .agents.base import Agent
from .cost import UnknownModel, cost_usd
from .grading.judge import JudgeError, LLMJudge
from .grading.safety import collect_safety_violations
from .exec import Environment, EnvironmentSpec
from .exec import attach as exec_attach
from .exec import harvest_into as exec_harvest
from .registry import ToolSession
from .scaffold import warnings as scaffold_warnings
from .tasks import LoadedTask
from .types import Check, RunResult, Score, Trajectory, Usage
from .state import World

ProgressFn = Callable[[str, RunResult | None], None]
#: Builds the agent for a given task. See `run_suite`.
AgentFactory = Callable[[LoadedTask], Agent]


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


def _agent_artifacts(world: World) -> list[dict[str, str]]:
    """Documents that reached the world other than through a tool call.

    Files named in `collect:` and anything a benchmark captures at grading time
    land here. Without this they are gradeable but unreadable: the report
    builds its artifact panel from the trajectory, so a container's whole
    output was missing from the one view meant to show what the agent produced.
    """
    try:
        documents = world.table("documents")
    except Exception:  # noqa: BLE001 - a world without documents is fine
        return []
    return [
        {"id": str(d.get("id", "")), "title": str(d.get("title", "")),
         "content": str(d.get("content", ""))}
        for d in documents
        if d.get("created_by") == "agent"
    ]


def _note(trajectory: Trajectory, message: str) -> None:
    """Append a failure to the trajectory without discarding earlier ones."""
    trajectory.error = (
        f"{trajectory.error + '; ' if trajectory.error else ''}{message}"
    )


async def run_one(
    task: LoadedTask, agent: Agent, config: RunConfig
) -> RunResult:
    """Execute and grade one run. Never raises.

    Every stage is contained on purpose. A run is a *measurement*, and a failed
    measurement is a data point — not grounds for discarding the fourteen that
    succeeded alongside it. Before this was total, an unpriced model id would
    let the agent run, spend real money, and then destroy the entire suite from
    inside the cost accounting.
    """
    agent_model = getattr(agent, "model", None)
    trajectory = Trajectory(task_id=task.id, agent=agent.name, model=agent_model)
    score = Score(w_state=config.w_state, w_rubric=config.w_rubric)
    status: str = "ok"
    started = time.time()

    def failed(message: str) -> RunResult:
        _note(trajectory, message)
        return RunResult(
            task_id=task.id, agent=agent.name, model=agent_model,
            trajectory=trajectory, score=score, status="harness_error",
            started_at=started,
        )

    # Setup: a malformed seed or a task naming an unknown tool is a task bug,
    # and should fail that task rather than the run that follows it.
    try:
        world = World(task.spec.seed)
        session = ToolSession(world, task.spec, trajectory)
    except Exception as exc:  # noqa: BLE001
        return failed(f"setup raised {type(exc).__name__}: {exc}")

    # Tasks with code execution get a container for the length of the run.
    # Provisioning can fail for reasons that have nothing to do with the agent
    # — no daemon, no image, a setup command that broke — so it is a harness
    # error rather than a score of zero.
    environment = None
    try:
        spec = EnvironmentSpec.from_config(task.spec.environment)
        if spec is not None:
            environment = Environment(spec)
            environment.start()
    except Exception as exc:  # noqa: BLE001
        return failed(f"environment setup raised {type(exc).__name__}: {exc}")
    exec_attach(world, environment)

    #: Checks produced inside the container, before it was destroyed.
    env_checks: list[Check] = []

    clock = time.perf_counter()
    try:
        await agent.run(task.spec, session, trajectory)
    except Exception as exc:  # noqa: BLE001 - a crashed agent is a result, not a stop
        status = "agent_error"
        _note(trajectory, f"{type(exc).__name__}: {exc}")
    finally:
        trajectory.wall_seconds = time.perf_counter() - clock
        if environment is not None:
            # Anything that needs the live container happens here, in this
            # order: grade first, then harvest, then destroy. Most real
            # benchmarks decide pass or fail by running the repository's own
            # tests, and neither an exit code nor a pytest summary is something
            # `collect:` can carry out of a container that no longer exists.
            if task.grade_in_environment is not None:
                try:
                    env_checks = task.grade_in_environment(
                        world, trajectory, environment
                    )
                except Exception as exc:  # noqa: BLE001
                    status = "harness_error"
                    _note(
                        trajectory,
                        f"in-container grading raised {type(exc).__name__}: {exc}",
                    )
            try:
                exec_harvest(world, environment)
            except Exception as exc:  # noqa: BLE001
                _note(trajectory, f"could not collect files: {exc}")
            trajectory.environment = environment.snapshot()
            environment.stop()
            exec_attach(world, None)

    # Grade whatever state the agent left behind, even if it crashed — a run
    # that errors after doing the work is a different failure from one that
    # errors having done nothing, and the checks are what tell them apart.
    try:
        score.state_checks = [*env_checks, *task.verify(world, trajectory)]
    except Exception as exc:  # noqa: BLE001 - a broken verifier is a harness bug
        status = "harness_error"
        score.state_checks = env_checks
        _note(trajectory, f"verifier raised {type(exc).__name__}: {exc}")

    try:
        score.safety_violations = collect_safety_violations(
            world, trajectory, session
        )
        if task.safety:
            score.safety_violations.extend(task.safety(world, trajectory))
    except Exception as exc:  # noqa: BLE001
        status = "harness_error"
        _note(trajectory, f"safety check raised {type(exc).__name__}: {exc}")

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
            _note(trajectory, str(exc))
        except Exception as exc:  # noqa: BLE001 - a judge bug is not a lost run
            status = "harness_error"
            _note(trajectory, f"judge raised {type(exc).__name__}: {exc}")

    # Pricing is a reporting concern and must never cost you the measurement.
    # An unknown model still reports zero loudly, via the status and the error.
    agent_cost = judge_cost = 0.0
    try:
        agent_cost = cost_usd(agent_model, trajectory.usage)
        # Priced against the judge's own model: it is frequently a different
        # one from the agent's, and with a local agent it is the only spend.
        judge_cost = cost_usd(judge_model, judge_usage)
    except UnknownModel as exc:
        status = "harness_error"
        _note(trajectory, f"cost not computed: {exc}")

    return RunResult(
        task_id=task.id,
        agent=agent.name,
        model=agent_model,
        trajectory=trajectory,
        score=score,
        artifacts=_agent_artifacts(world),
        warnings=scaffold_warnings(task.spec, trajectory),
        agent_cost_usd=agent_cost,
        judge_cost_usd=judge_cost,
        judge_model=judge_model,
        judge_usage=judge_usage,
        status=status,  # type: ignore[arg-type]
        started_at=started,
    )


async def run_suite(
    tasks: list[LoadedTask],
    agent: Agent | AgentFactory,
    config: RunConfig | None = None,
    on_progress: ProgressFn | None = None,
    completed: dict[str, int] | None = None,
) -> list[RunResult]:
    """Run every task, `config.repeats` times each, up to `config.concurrency`.

    `agent` is normally one agent used for every task. A callable is accepted
    for the case where the agent depends on the task — replaying each task's
    own reference trajectory, most obviously. That form exists so `--gold` is
    an ordinary suite run rather than a second, sequential code path that
    quietly ignored concurrency and drifted from this one.

    `completed` counts runs an earlier attempt already finished, per task, and
    is how a resumed suite skips them. Counted rather than listed because
    repeats of one task are interchangeable measurements — the fourth run of a
    task is not a distinct thing to be matched up, it is one more sample.
    """
    config = config or RunConfig()
    semaphore = asyncio.Semaphore(config.concurrency)
    build = agent if callable(agent) and not hasattr(agent, "run") else None

    async def guarded(task: LoadedTask) -> RunResult:
        async with semaphore:
            if on_progress:
                on_progress(task.id, None)
            result = await run_one(task, build(task) if build else agent, config)
            if on_progress:
                on_progress(task.id, result)
            return result

    remaining = dict(completed or {})
    scheduled: list[LoadedTask] = []
    for task in tasks:
        already = remaining.get(task.id, 0)
        scheduled += [task] * max(0, config.repeats - already)
    jobs: list[Awaitable[RunResult]] = [guarded(task) for task in scheduled]
    # `run_one` is total, so nothing should arrive here as an exception. This
    # is the backstop for the case where something does anyway — including a
    # bug in the progress callback. Losing a whole paid suite to one unexpected
    # traceback is the failure mode being ruled out.
    settled = await asyncio.gather(*jobs, return_exceptions=True)

    results: list[RunResult] = []
    for task, outcome in zip(scheduled, settled):
        if isinstance(outcome, BaseException):
            trajectory = Trajectory(task_id=task.id, agent=agent.name)
            trajectory.error = (
                f"run escaped containment: {type(outcome).__name__}: {outcome}"
            )
            results.append(
                RunResult(
                    task_id=task.id,
                    agent=agent.name,
                    model=getattr(agent, "model", None),
                    trajectory=trajectory,
                    score=Score(w_state=config.w_state, w_rubric=config.w_rubric),
                    status="harness_error",
                )
            )
        else:
            results.append(outcome)
    return results
