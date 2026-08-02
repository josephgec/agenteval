"""Scoring arithmetic, safety gating, and cost accounting."""

from datetime import date

import pytest

from agenteval import Check, Score, Usage, cost_usd
from agenteval.cost import UnknownModel
from agenteval.types import RubricScore


def test_state_score_is_weighted():
    score = Score(
        state_checks=[
            Check("a", passed=True, weight=3.0),
            Check("b", passed=False, weight=1.0),
        ]
    )
    assert score.state_score == 0.75


def test_overall_blends_state_and_rubric():
    score = Score(
        state_checks=[Check("a", passed=True)],
        rubric_scores=[RubricScore("r", score=0.5, weight=1.0, reasoning="")],
        w_state=0.7,
        w_rubric=0.3,
    )
    assert score.overall == pytest.approx(0.7 * 1.0 + 0.3 * 0.5)


def test_overall_falls_back_to_state_when_there_is_no_rubric():
    score = Score(state_checks=[Check("a", passed=True), Check("b", passed=False)])
    assert score.rubric_score is None
    assert score.overall == 0.5


def test_a_safety_violation_zeroes_an_otherwise_perfect_run():
    """Reaching the right end state by a forbidden route is not a pass."""
    score = Score(
        state_checks=[Check("a", passed=True)],
        rubric_scores=[RubricScore("r", score=1.0, weight=1.0, reasoning="")],
        safety_violations=["attempted admin.delete_record"],
    )
    assert score.state_score == 1.0
    assert score.rubric_score == 1.0
    assert score.safe is False
    assert score.overall == 0.0


def test_empty_score_is_zero_not_an_error():
    assert Score().overall == 0.0


def test_zero_weight_checks_do_not_divide_by_zero():
    """A task author can legitimately zero a check out while iterating."""
    score = Score(state_checks=[Check("a", passed=True, weight=0.0)])
    assert score.state_score is None
    assert score.overall == 0.0


def test_zero_weight_rubric_does_not_divide_by_zero():
    score = Score(
        state_checks=[Check("a", passed=True)],
        rubric_scores=[RubricScore("r", score=1.0, weight=0.0, reasoning="")],
    )
    assert score.rubric_score is None
    assert score.overall == 1.0  # falls back to state alone


# -- trajectory helpers used by verifiers ----------------------------------- #


def test_calls_to_ignores_blocked_calls():
    """A verifier asking 'did they read the policy' must not be satisfied by an
    attempt that the harness rejected."""
    from agenteval import Trajectory
    from agenteval.types import ToolCall

    trajectory = Trajectory(task_id="t", agent="a")
    trajectory.calls = [
        ToolCall(step=1, name="docs_read", input={"doc_id": "p"}, output="ok"),
        ToolCall(step=2, name="docs_read", input={}, output="err",
                 blocked_reason="bad_args"),
    ]
    assert len(trajectory.calls_to("docs_read")) == 1
    assert trajectory.called("docs_read") is True
    assert trajectory.called("docs_write") is False
    assert trajectory.steps == 2  # blocked calls still count against budget


# -- cost ------------------------------------------------------------------- #


def test_cost_uses_published_rates():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd("claude-opus-5", usage, on=date(2026, 8, 1)) == pytest.approx(30.0)


def test_cache_reads_and_writes_are_priced_off_the_input_rate():
    usage = Usage(cache_read_input_tokens=1_000_000)
    assert cost_usd("claude-opus-5", usage, on=date(2026, 8, 1)) == pytest.approx(0.5)
    usage = Usage(cache_creation_input_tokens=1_000_000)
    assert cost_usd("claude-opus-5", usage, on=date(2026, 8, 1)) == pytest.approx(6.25)


def test_introductory_pricing_applies_only_inside_its_window():
    usage = Usage(input_tokens=1_000_000)
    assert cost_usd("claude-sonnet-5", usage, on=date(2026, 8, 15)) == pytest.approx(2.0)
    assert cost_usd("claude-sonnet-5", usage, on=date(2026, 9, 1)) == pytest.approx(3.0)


def test_an_unpriced_model_raises_rather_than_costing_zero():
    with pytest.raises(UnknownModel):
        cost_usd("claude-not-a-model", Usage(input_tokens=10))


def test_a_modelless_agent_costs_nothing():
    assert cost_usd(None, Usage(input_tokens=10_000)) == 0.0
