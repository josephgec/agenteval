"""Runner orchestration: isolation, error containment, and status reporting.

Built on ad-hoc tasks rather than the real suite so each property is isolated
and the assertions say exactly what they mean.
"""

import asyncio

import pytest

from agenteval import (
    Check,
    RunConfig,
    ScriptedAgent,
    TaskSpec,
    Trajectory,
    Usage,
    run_one,
    run_suite,
)
from agenteval.grading.judge import JudgeError, JudgeOutcome
from agenteval.tasks import LoadedTask
from agenteval.types import RubricCriterion, RubricScore

SEED = {
    "tickets": [
        {"id": "TKT-1", "subject": "s", "body": "b", "status": "open",
         "priority": "P3", "team": "support", "comments": []}
    ]
}


def make_task(verify=None, safety=None, rubric=None, **spec_kwargs):
    spec = TaskSpec(
        id="t",
        prompt="p",
        seed=SEED,
        rubric=rubric or [],
        rubric_artifacts=["final_text"] if rubric else [],
        **spec_kwargs,
    )
    return LoadedTask(
        spec=spec,
        verify=verify or (lambda w, t: [Check("ok", passed=True)]),
        safety=safety,
        gold=None,
    )


class ExplodingAgent:
    name, model = "boom", None

    def __init__(self, script=()):
        self.script = list(script)

    async def run(self, task, session, trajectory):
        for step in self.script:
            session.call(step["tool"], step.get("input", {}))
        raise RuntimeError("scaffold crashed")


class StubJudge:
    def __init__(self, scores=None, raises=None, usage=None,
                 model="claude-opus-5"):
        self.scores = scores or []
        self.raises = raises
        self.usage = usage or Usage()
        self.model = model
        self.calls = 0

    async def score(self, task, world, trajectory):
        self.calls += 1
        if self.raises:
            raise self.raises
        return JudgeOutcome(scores=self.scores, usage=self.usage,
                            model=self.model)


# --------------------------------------------------------------------------- #
# Isolation
# --------------------------------------------------------------------------- #


async def test_each_run_gets_its_own_world():
    """Repeats must not inherit the previous run's mutations."""
    task = make_task(
        verify=lambda w, t: [
            Check("still P3", passed=w.find("tickets", "TKT-1")["priority"] == "P3")
        ]
    )
    script = [
        {"tool": "tickets_update", "input": {"ticket_id": "TKT-1", "priority": "P0"}}
    ]
    results = await run_suite(
        [task], ScriptedAgent(script), RunConfig(repeats=3, concurrency=3)
    )
    # Every run starts at P3 and sets P0; none of them observe P0 at the start.
    assert len(results) == 3
    assert all(r.score.state_score == 0.0 for r in results)


async def test_the_seed_itself_is_never_mutated():
    task = make_task()
    script = [
        {"tool": "tickets_update", "input": {"ticket_id": "TKT-1", "priority": "P0"}}
    ]
    await run_one(task, ScriptedAgent(script), RunConfig())
    assert task.spec.seed["tickets"][0]["priority"] == "P3"


async def test_concurrency_limit_is_respected():
    in_flight = peak = 0

    class SlowAgent:
        name, model = "slow", None

        async def run(self, task, session, trajectory):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1

    await run_suite([make_task()], SlowAgent(), RunConfig(repeats=8, concurrency=2))
    assert peak <= 2


# --------------------------------------------------------------------------- #
# Error containment
# --------------------------------------------------------------------------- #


async def test_a_crashed_agent_is_a_result_not_a_stop():
    result = await run_one(make_task(), ExplodingAgent(), RunConfig())
    assert result.status == "agent_error"
    assert "RuntimeError: scaffold crashed" in result.trajectory.error


async def test_work_done_before_a_crash_is_still_graded():
    """A run that errors *after* doing the work is a different failure from one
    that errors having done nothing, and only the checks can tell them apart."""
    task = make_task(
        verify=lambda w, t: [
            Check("priority set", passed=w.find("tickets", "TKT-1")["priority"] == "P0")
        ]
    )
    agent = ExplodingAgent(
        [{"tool": "tickets_update",
          "input": {"ticket_id": "TKT-1", "priority": "P0"}}]
    )
    result = await run_one(task, agent, RunConfig())
    assert result.status == "agent_error"
    assert result.score.state_score == 1.0  # the work still counted


async def test_one_crashed_run_does_not_take_down_the_suite():
    results = await run_suite(
        [make_task(), make_task()], ExplodingAgent(), RunConfig(repeats=2)
    )
    assert len(results) == 4
    assert all(r.status == "agent_error" for r in results)


# A run is a measurement. A failed measurement is a data point, not grounds for
# discarding the ones that succeeded beside it — especially when they were paid
# for. Every stage below used to be able to destroy a whole suite.


async def test_an_unpriced_model_does_not_destroy_the_suite():
    """The regression that motivated making run_one total: the agent runs and
    spends real money, then cost accounting raises and takes every result with
    it. Guaranteed to happen the day a new model ships.
    """
    class FutureModelAgent:
        name, model = "future", "claude-opus-99"  # not in PRICING

        async def run(self, task, session, trajectory):
            trajectory.usage.add(Usage(input_tokens=500_000))

    results = await run_suite(
        [make_task(), make_task()], FutureModelAgent(), RunConfig(repeats=2)
    )
    assert len(results) == 4
    assert all(r.status == "harness_error" for r in results)
    assert all("claude-opus-99" in r.trajectory.error for r in results)
    # The measurement survives even though the price does not.
    assert all(r.score.state_score == 1.0 for r in results)
    assert all(r.cost_usd == 0.0 for r in results)


async def test_a_malformed_seed_fails_only_its_own_task():
    broken = make_task()
    broken.spec.seed = {"acounts": []}  # typo in a task fixture
    results = await run_suite(
        [broken, make_task()], ScriptedAgent([]), RunConfig()
    )
    statuses = sorted(r.status for r in results)
    assert statuses == ["harness_error", "ok"]
    assert "setup raised" in next(
        r.trajectory.error for r in results if r.status == "harness_error"
    )


async def test_a_task_naming_an_unknown_tool_fails_only_itself():
    results = await run_suite(
        [make_task(allowed_tools=["not_a_tool"]), make_task()],
        ScriptedAgent([]),
        RunConfig(),
    )
    assert sorted(r.status for r in results) == ["harness_error", "ok"]


async def test_a_broken_safety_function_is_contained():
    """The verifier was guarded but its sibling wasn't — an inconsistency that
    let a one-line task bug lose the suite."""
    def exploding_safety(world, trajectory):
        raise RuntimeError("bad safety rule")

    result = await run_one(make_task(safety=exploding_safety),
                           ScriptedAgent([]), RunConfig())
    assert result.status == "harness_error"
    assert "safety check raised RuntimeError" in result.trajectory.error
    assert result.score.state_score == 1.0  # grading still happened


async def test_a_judge_raising_something_unexpected_is_contained():
    """JudgeError was handled; an ordinary bug in a custom judge was not."""
    class BuggyJudge:
        model = "claude-opus-5"

        async def score(self, task, world, trajectory):
            raise AttributeError("typo in a custom judge")

    task = make_task(rubric=[RubricCriterion(id="a", description="d")])
    result = await run_one(task, ScriptedAgent([]), RunConfig(judge=BuggyJudge()))
    assert result.status == "harness_error"
    assert "judge raised AttributeError" in result.trajectory.error


async def test_a_crashing_progress_callback_does_not_lose_results():
    def exploding_progress(task_id, result):
        raise RuntimeError("bad reporter")

    results = await run_suite(
        [make_task()], ScriptedAgent([]), RunConfig(repeats=2),
        on_progress=exploding_progress,
    )
    assert len(results) == 2
    assert all("escaped containment" in r.trajectory.error for r in results)


async def test_a_broken_verifier_is_reported_as_a_harness_error():
    def exploding_verify(world, trajectory):
        raise KeyError("bad_field")

    result = await run_one(make_task(verify=exploding_verify),
                           ScriptedAgent([]), RunConfig())
    assert result.status == "harness_error"
    assert "verifier raised KeyError" in result.trajectory.error
    assert result.score.state_checks == []


async def test_agent_and_verifier_errors_are_both_reported():
    def exploding_verify(world, trajectory):
        raise ValueError("nope")

    result = await run_one(make_task(verify=exploding_verify),
                           ExplodingAgent(), RunConfig())
    assert "scaffold crashed" in result.trajectory.error
    assert "verifier raised ValueError" in result.trajectory.error


# --------------------------------------------------------------------------- #
# Judge integration
# --------------------------------------------------------------------------- #


async def test_a_judge_failure_is_surfaced_rather_than_degrading_the_score():
    """Silently returning no rubric would leave the run scored on state alone —
    a higher number than it earned."""
    task = make_task(rubric=[RubricCriterion(id="a", description="d")])
    judge = StubJudge(raises=JudgeError("judge refused to grade this task"))
    result = await run_one(task, ScriptedAgent([]),
                           RunConfig(judge=judge))
    assert result.status == "harness_error"
    assert "judge refused" in result.trajectory.error
    assert result.score.rubric_scores == []


async def test_the_judge_is_skipped_when_the_task_has_no_rubric():
    judge = StubJudge()
    await run_one(make_task(), ScriptedAgent([]), RunConfig(judge=judge))
    assert judge.calls == 0


async def test_rubric_scores_reach_the_result():
    task = make_task(rubric=[RubricCriterion(id="a", description="d", weight=1.0)])
    judge = StubJudge([RubricScore("a", score=0.5, weight=1.0, reasoning="ok")])
    result = await run_one(task, ScriptedAgent([]), RunConfig(judge=judge))
    assert result.score.rubric_score == 0.5
    assert result.status == "ok"


async def test_configured_weights_reach_the_score():
    task = make_task(rubric=[RubricCriterion(id="a", description="d")])
    judge = StubJudge([RubricScore("a", score=0.0, weight=1.0, reasoning="")])
    result = await run_one(
        task, ScriptedAgent([]), RunConfig(judge=judge, w_state=0.9, w_rubric=0.1)
    )
    assert result.score.overall == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Safety and accounting
# --------------------------------------------------------------------------- #


async def test_task_specific_safety_merges_with_the_built_in_checks():
    task = make_task(
        safety=lambda w, t: ["task-specific violation"],
        forbidden_tools=["admin_delete_record"],
    )
    script = [
        {"tool": "admin_delete_record",
         "input": {"collection": "tickets", "record_id": "TKT-1"}}
    ]
    result = await run_one(task, ScriptedAgent(script), RunConfig())
    joined = " ".join(result.score.safety_violations)
    assert "task-specific violation" in joined      # from the task
    assert "admin_delete_record" in joined          # from the harness
    assert result.score.overall == 0.0


class BillingAgent:
    name, model = "billing", "claude-opus-5"

    async def run(self, task, session, trajectory):
        trajectory.usage.add(Usage(input_tokens=1_000_000, output_tokens=1_000_000))


async def test_cost_is_derived_from_recorded_usage():
    result = await run_one(make_task(), BillingAgent(), RunConfig())
    assert result.agent_cost_usd == pytest.approx(30.0)
    assert result.cost_usd == pytest.approx(30.0)


async def test_a_modelless_agent_reports_no_cost():
    result = await run_one(make_task(), ScriptedAgent([]), RunConfig())
    assert result.cost_usd == 0.0
    assert result.model is None


async def test_the_judge_is_billed_against_its_own_model():
    """Agent and judge are frequently different models on different pricing."""
    task = make_task(rubric=[RubricCriterion(id="a", description="d")])
    judge = StubJudge(
        [RubricScore("a", score=1.0, weight=1.0, reasoning="")],
        usage=Usage(input_tokens=1_000_000, output_tokens=1_000_000),
        model="claude-haiku-4-5",  # $1/$5 rather than the agent's $5/$25
    )
    result = await run_one(task, BillingAgent(), RunConfig(judge=judge))

    assert result.agent_cost_usd == pytest.approx(30.0)
    assert result.judge_cost_usd == pytest.approx(6.0)
    assert result.cost_usd == pytest.approx(36.0)
    assert result.judge_model == "claude-haiku-4-5"


async def test_a_local_agent_still_reports_the_judges_spend():
    """The bug this replaced: a free agent graded by a hosted judge reported
    $0.00 for a run that genuinely cost money."""
    task = make_task(rubric=[RubricCriterion(id="a", description="d")])
    judge = StubJudge(
        [RubricScore("a", score=1.0, weight=1.0, reasoning="")],
        usage=Usage(input_tokens=1_000_000),
    )
    result = await run_one(task, ScriptedAgent([]), RunConfig(judge=judge))

    assert result.agent_cost_usd == 0.0
    assert result.judge_cost_usd == pytest.approx(5.0)
    assert result.cost_usd == pytest.approx(5.0)


async def test_a_failed_judge_call_is_still_billed():
    task = make_task(rubric=[RubricCriterion(id="a", description="d")])
    judge = StubJudge(
        raises=JudgeError("judge refused", Usage(input_tokens=1_000_000))
    )
    result = await run_one(task, ScriptedAgent([]), RunConfig(judge=judge))

    assert result.status == "harness_error"
    assert result.judge_cost_usd == pytest.approx(5.0)


async def test_an_unjudged_run_carries_no_judge_cost():
    result = await run_one(make_task(), ScriptedAgent([]),
                           RunConfig(judge=StubJudge()))
    assert result.judge_cost_usd == 0.0
    assert result.judge_usage.input_tokens == 0


async def test_wall_time_is_measured():
    class SleepyAgent:
        name, model = "sleepy", None

        async def run(self, task, session, trajectory):
            await asyncio.sleep(0.02)

    result = await run_one(make_task(), SleepyAgent(), RunConfig())
    assert result.trajectory.wall_seconds >= 0.02


# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #


async def test_progress_fires_before_and_after_each_run():
    events = []

    def on_progress(task_id, result):
        events.append((task_id, result is None))

    await run_suite([make_task()], ScriptedAgent([]), RunConfig(repeats=2),
                    on_progress=on_progress)
    assert events.count(("t", True)) == 2   # start
    assert events.count(("t", False)) == 2  # finish
