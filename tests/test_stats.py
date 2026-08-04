"""How much of a difference is a difference.

The point of every test here is that a number should not look more meaningful
than it is. Most of these assert on the *width* of an interval or on a refusal
to call a winner, which is the behaviour that stops a twenty-instance sample
being read as a ranking.
"""

import pytest

from agenteval import report as report_mod
from agenteval import stats


# --------------------------------------------------------------------------- #
# Intervals
# --------------------------------------------------------------------------- #


def test_twenty_instances_is_a_very_wide_band():
    """The finding that prompted all of this: pass@1 of 30% at n=20 spans more
    than thirty points, so 30% and 45% on that sample are the same claim."""
    band = stats.wilson(6, 20)
    assert band.point == pytest.approx(0.30)
    assert band.width > 0.30
    assert band.low < 0.45 < band.high  # 45% is inside the interval for 30%


def test_a_perfect_score_does_not_claim_certainty():
    """Where the textbook normal approximation breaks: at 20 for 20 it puts the
    lower bound at 100%, asserting certainty from twenty observations."""
    band = stats.wilson(20, 20)
    assert band.point == 1.0
    assert band.low < 0.9 and band.high == 1.0


def test_a_zero_score_does_not_go_negative():
    band = stats.wilson(0, 20)
    assert band.low == 0.0 and 0.0 < band.high < 0.25


def test_more_data_narrows_the_band():
    assert stats.wilson(200, 400).width < stats.wilson(10, 20).width


def test_no_data_is_not_a_crash():
    assert stats.wilson(0, 0).n == 0
    assert stats.interval([]).point == 0.0
    assert stats.bootstrap([]).n == 0


# --------------------------------------------------------------------------- #
# Choosing the right interval
# --------------------------------------------------------------------------- #


def test_pass_fail_scores_get_a_wilson_interval():
    scores = [1.0, 0.0, 1.0, 1.0, 0.0]
    assert stats.is_binary(scores)
    assert stats.interval(scores) == stats.wilson(3, 5)


def test_weighted_scores_get_a_bootstrap_instead():
    """The enterprise tasks produce blends of weighted checks, not proportions.
    Treating them as coin flips would be the wrong model."""
    scores = [0.9, 0.85, 1.0, 0.7]
    assert not stats.is_binary(scores)
    assert stats.interval(scores) == stats.bootstrap(scores)


def test_the_bootstrap_is_reproducible():
    """A confidence bound that flickers between runs of the reporting code is
    worse than none."""
    scores = [0.9, 0.2, 0.55, 0.7, 0.31]
    assert stats.bootstrap(scores).render() == stats.bootstrap(scores).render()


def test_a_single_observation_admits_it_knows_nothing():
    assert stats.bootstrap([0.5]).width == 1.0


def test_the_interval_brackets_the_mean():
    scores = [0.9, 0.2, 0.55, 0.7, 0.31]
    band = stats.bootstrap(scores)
    assert band.low <= band.point <= band.high


# --------------------------------------------------------------------------- #
# Comparing two models
# --------------------------------------------------------------------------- #


def _spread(values):
    return {f"t{i}": [v] for i, v in enumerate(values)}


def test_a_clear_difference_is_called():
    left = _spread([0.0] * 20)
    right = _spread([1.0] * 20)
    outcome = stats.compare(left, right, "weak", "strong")
    assert outcome.decisive
    assert "strong is ahead" in outcome.verdict()
    assert outcome.wins == 20 and outcome.losses == 0


def test_a_difference_the_sample_cannot_support_is_not_called():
    """The behaviour this module exists for. Two models a few points apart over
    twenty instances have not been ranked."""
    left = _spread([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 2)
    right = _spread([1, 1, 1, 0, 0, 0, 1, 0, 1, 0] * 2)
    outcome = stats.compare(left, right)
    assert not outcome.decisive
    assert "sign is not established" in outcome.verdict()


def test_pairing_uses_only_shared_instances():
    """A model that ran different instances has not been compared to anything;
    including its marginal mean would be comparing different exams."""
    outcome = stats.compare(
        {"a": [1.0], "b": [1.0], "c": [1.0], "d": [1.0], "e": [1.0], "x": [0.0]},
        {"a": [0.0], "b": [0.0], "c": [0.0], "d": [0.0], "e": [0.0], "y": [1.0]},
    )
    assert outcome.paired == 5
    assert outcome.unpaired == 2


def test_too_few_shared_instances_refuses_to_compare():
    """Two instances that both improved is a fact about two instances. A
    percentile bootstrap over them will happily exclude zero while establishing
    nothing."""
    outcome = stats.compare(_spread([0.0, 0.0]), _spread([1.0, 1.0]))
    assert not outcome.decisive
    assert "too few to compare" in outcome.verdict()


def test_nothing_in_common_says_so():
    outcome = stats.compare({"a": [1.0]}, {"b": [1.0]})
    assert outcome.paired == 0
    assert "nothing to compare" in outcome.verdict()


def test_the_direction_is_reported_correctly():
    ahead_on_the_left = stats.compare(
        _spread([1.0] * 10), _spread([0.0] * 10), "left", "right"
    )
    assert "left is ahead" in ahead_on_the_left.verdict()
    assert ahead_on_the_left.difference.point < 0


def test_repeats_of_an_instance_average_before_differencing():
    outcome = stats.compare(
        {f"t{i}": [0.0, 1.0] for i in range(10)},
        {f"t{i}": [1.0, 1.0] for i in range(10)},
    )
    assert outcome.difference.point == pytest.approx(0.5)


def test_pairing_sees_a_difference_that_marginal_means_would_miss():
    """The reason for pairing. Instance difficulty dominates: five easy and
    five hard problems, with one model better on every single one. The spread
    between instances swamps the spread between models, so unpaired the two
    means look close — paired, every instance points the same way.
    """
    easy, hard = [0.9] * 5, [0.1] * 5
    left = _spread(easy + hard)
    right = _spread([e + 0.08 for e in easy] + [h + 0.08 for h in hard])
    outcome = stats.compare(left, right)
    assert outcome.wins == 10 and outcome.losses == 0
    assert outcome.decisive  # despite the means being only 0.08 apart


# --------------------------------------------------------------------------- #
# How much would settle it
# --------------------------------------------------------------------------- #


def test_smaller_differences_need_more_data():
    assert stats.sample_size_for(0.05) > stats.sample_size_for(0.20)


def test_detecting_ten_points_needs_more_than_swe_bench_lite_has():
    """Worth knowing before quoting a ten-point gap on a 300-instance split."""
    assert stats.sample_size_for(0.10) > 300


def test_a_nonsense_difference_asks_for_nothing():
    assert stats.sample_size_for(0.0) == 0
    assert stats.sample_size_for(-0.1) == 0


# --------------------------------------------------------------------------- #
# Where it surfaces
# --------------------------------------------------------------------------- #


def _run_result(task_id, score):
    from agenteval.types import Check, RunResult, Score, Trajectory

    return RunResult(
        task_id=task_id, agent="a", model=None,
        trajectory=Trajectory(task_id, "a"),
        score=Score(state_checks=[Check("c", passed=bool(score))]),
    )


def test_the_run_summary_carries_an_interval(capsys):
    from rich.console import Console

    results = [_run_result(f"t{i}", i % 3 == 0) for i in range(20)]
    report_mod.print_results(results, Console())
    out = capsys.readouterr().out
    assert "95%" in out
    # And says plainly how much data a real comparison would need.
    assert "instances; you ran 20" in out


def test_a_narrow_result_is_not_lectured_about(capsys):
    from rich.console import Console

    report_mod.print_results([_run_result(f"t{i}", True) for i in range(400)],
                             Console())
    assert "would need about" not in capsys.readouterr().out


def _payload(agent, scores):
    return {
        "meta": {"agent": agent, "total_cost_usd": 0.0},
        "results": [
            {"task_id": task, "score": {"overall": score}}
            for task, score in scores.items()
        ],
    }


def test_comparing_two_runs_reports_the_paired_difference(capsys):
    from rich.console import Console

    left = _payload("weak", {f"t{i}": 0.0 for i in range(20)})
    right = _payload("strong", {f"t{i}": 1.0 for i in range(20)})
    report_mod.print_comparison(left, right, Console())
    out = capsys.readouterr().out
    assert "strong is ahead" in out
    assert "paired difference" in out


def test_comparing_says_when_it_cannot_tell(capsys):
    from rich.console import Console

    left = _payload("a", {f"t{i}": float(i % 2) for i in range(20)})
    right = _payload("b", {f"t{i}": float((i + 1) % 2) for i in range(20)})
    report_mod.print_comparison(left, right, Console())
    out = capsys.readouterr().out
    assert "not established" in out
    assert "shared instances to establish" in out


def test_comparing_excludes_instances_only_one_side_ran(capsys):
    from rich.console import Console

    left = _payload("a", {"shared": 1.0, "only-left": 1.0})
    right = _payload("b", {"shared": 0.0, "only-right": 0.0})
    report_mod.print_comparison(left, right, Console())
    assert "run by only one side" in capsys.readouterr().out
