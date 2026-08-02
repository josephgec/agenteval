"""Aggregation, persistence, and rendering.

`aggregate()` produces every number a person actually reads. If its mean is
wrong, nothing downstream notices — there is no exception, just a confidently
wrong table — so it gets tested directly rather than through the CLI.
"""

import json

import pytest
from rich.console import Console

from agenteval import Check, RunResult, Score, Trajectory, Usage
from agenteval import report as report_mod
from agenteval.types import RubricScore, ToolCall


def make_result(
    task_id="t",
    overall_checks=(True,),
    rubric=None,
    violations=(),
    steps=0,
    cost=0.0,
    status="ok",
    agent="a",
    seconds=1.0,
):
    trajectory = Trajectory(task_id=task_id, agent=agent, model="claude-opus-5")
    trajectory.wall_seconds = seconds
    trajectory.calls = [
        ToolCall(step=i + 1, name="tickets_get", input={}, output="ok")
        for i in range(steps)
    ]
    score = Score(
        state_checks=[Check(f"c{i}", passed=p) for i, p in enumerate(overall_checks)],
        rubric_scores=list(rubric or []),
        safety_violations=list(violations),
    )
    return RunResult(
        task_id=task_id,
        agent=agent,
        model="claude-opus-5",
        trajectory=trajectory,
        score=score,
        agent_cost_usd=cost,
        status=status,
    )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_repeats_collapse_into_mean_and_spread():
    results = [
        make_result("t", overall_checks=(True,)),   # 1.0
        make_result("t", overall_checks=(False,)),  # 0.0
    ]
    stats = report_mod.aggregate(results)["t"]
    assert stats["n"] == 2
    assert stats["overall_mean"] == pytest.approx(0.5)
    # Spread is the point of repeats: 1.0 then 0.0 is an unreliable agent,
    # not a "0.5 agent", and the table has to be able to say so.
    assert stats["overall_stdev"] == pytest.approx(0.7071, abs=1e-3)


def test_single_run_reports_zero_spread_rather_than_erroring():
    """statistics.stdev raises on n=1; the mean is still meaningful."""
    stats = report_mod.aggregate([make_result("t")])["t"]
    assert stats["n"] == 1
    assert stats["overall_stdev"] == 0.0


def test_tasks_are_grouped_independently():
    stats = report_mod.aggregate(
        [
            make_result("a", overall_checks=(True,)),
            make_result("b", overall_checks=(False,)),
            make_result("a", overall_checks=(True,)),
        ]
    )
    assert stats["a"]["n"] == 2 and stats["a"]["overall_mean"] == 1.0
    assert stats["b"]["n"] == 1 and stats["b"]["overall_mean"] == 0.0


def test_rubric_mean_is_none_when_no_run_was_judged():
    """Distinguishes 'not judged' from 'judged and scored zero'."""
    assert report_mod.aggregate([make_result("t")])["t"]["rubric_mean"] is None


def test_rubric_mean_averages_only_judged_runs():
    judged = make_result(
        "t", rubric=[RubricScore("r", score=0.5, weight=1.0, reasoning="")]
    )
    unjudged = make_result("t")
    stats = report_mod.aggregate([judged, unjudged])["t"]
    assert stats["rubric_mean"] == pytest.approx(0.5)


def test_unsafe_and_errored_runs_are_counted_separately():
    stats = report_mod.aggregate(
        [
            make_result("t", violations=["reached for admin.delete_record"]),
            make_result("t", status="agent_error"),
            make_result("t"),
        ]
    )["t"]
    assert stats["unsafe_runs"] == 1
    assert stats["errors"] == 1


def test_cost_totals_and_process_means():
    stats = report_mod.aggregate(
        [
            make_result("t", steps=4, cost=0.10, seconds=10.0),
            make_result("t", steps=8, cost=0.30, seconds=20.0),
        ]
    )["t"]
    assert stats["cost_total"] == pytest.approx(0.40)  # summed, not averaged
    assert stats["steps_mean"] == 6.0                  # averaged, not summed
    assert stats["seconds_mean"] == 15.0


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_save_load_round_trip_preserves_failure_detail(tmp_path):
    """The JSON is the durable record; a lost `detail` means an unexplainable
    failure a week later."""
    result = make_result("t", overall_checks=(False,), steps=3, cost=0.25)
    result.score.state_checks[0].detail = "expected 'closed', got 'open'"
    result.score.rubric_scores = [
        RubricScore("tone", score=0.5, weight=2.0, reasoning="quoted span")
    ]
    result.score.safety_violations = ["attempted admin.delete_record"]

    path = report_mod.save([result], tmp_path / "run", {"agent": "claude"})
    payload = report_mod.load(path)

    [saved] = payload["results"]
    assert saved["score"]["checks"][0]["detail"] == "expected 'closed', got 'open'"
    assert saved["score"]["rubric_scores"][0]["reasoning"] == "quoted span"
    assert saved["score"]["safety_violations"] == ["attempted admin.delete_record"]
    assert saved["process"]["steps"] == 3
    assert payload["meta"]["agent"] == "claude"
    assert payload["meta"]["total_cost_usd"] == pytest.approx(0.25)


def test_load_accepts_a_directory_or_the_file_itself(tmp_path):
    report_mod.save([make_result()], tmp_path / "run", {"agent": "a"})
    from_dir = report_mod.load(tmp_path / "run")
    from_file = report_mod.load(tmp_path / "run" / "results.json")
    assert from_dir == from_file


def test_saved_payload_is_valid_json_for_external_tooling(tmp_path):
    path = report_mod.save([make_result()], tmp_path / "run", {"agent": "a"})
    json.loads(path.read_text())  # raises if the dataclasses leaked through


def test_save_creates_nested_directories(tmp_path):
    path = report_mod.save([make_result()], tmp_path / "a" / "b", {"agent": "x"})
    assert path.exists()


def test_empty_results_do_not_crash_the_summary(tmp_path):
    payload = report_mod.load(report_mod.save([], tmp_path / "run", {"agent": "a"}))
    assert payload["meta"]["runs"] == 0
    assert payload["meta"]["mean_overall"] == 0.0


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #


def test_html_report_lists_failures_not_just_scores(tmp_path):
    result = make_result("expense_approval", overall_checks=(False,))
    result.score.state_checks[0].detail = "expected escalation to EMP-003"
    path = report_mod.write_html([result], tmp_path, {"agent": "claude:opus-5"})
    page = path.read_text()
    assert "expense_approval" in page
    assert "expected escalation to EMP-003" in page
    assert "claude:opus-5" in page


def test_html_report_escapes_agent_authored_text(tmp_path):
    """The judge quotes spans from the artifacts, so its reasoning carries
    model-authored text into a page intended for sharing."""
    result = make_result("t", overall_checks=(False,))
    result.score.rubric_scores = [
        RubricScore(
            "tone",
            score=0.0,
            weight=1.0,
            reasoning="the email said </span><script>alert(1)</script>",
        )
    ]
    result.score.state_checks[0].detail = "<img src=x onerror=alert(2)>"
    result.score.safety_violations = ["<b>injected</b>"]

    page = report_mod.write_html([result], tmp_path, {"agent": "a"}).read_text()

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
    assert "<img src=x onerror=alert(2)>" not in page
    assert "<b>injected</b>" not in page


def test_html_report_says_so_when_there_is_nothing_to_report(tmp_path):
    page = report_mod.write_html([make_result()], tmp_path, {"agent": "a"}).read_text()
    assert "No findings." in page


def test_html_report_is_self_contained(tmp_path):
    """No external fetches: the CSP on most sharing surfaces blocks them, and a
    report that renders differently for the reader is worse than none."""
    page = report_mod.write_html([make_result()], tmp_path, {"agent": "a"}).read_text()
    for external in ("http://", "https://", "<script"):
        assert external not in page


# --------------------------------------------------------------------------- #
# Terminal and comparison
# --------------------------------------------------------------------------- #


def render(fn, *args) -> str:
    console = Console(record=True, width=200, force_terminal=False)
    fn(*args, console)
    return console.export_text()


def test_terminal_output_leads_with_the_failed_check():
    result = make_result("expense_approval", overall_checks=(True, False))
    result.score.state_checks[1].name = "EXP-502 escalated to direct manager"
    result.score.state_checks[1].detail = "expected 'EMP-003', got None"
    text = render(report_mod.print_results, [result])
    assert "EXP-502 escalated to direct manager" in text
    assert "expected 'EMP-003', got None" in text


def test_terminal_output_surfaces_safety_violations():
    result = make_result("t", violations=["attempted admin.delete_record at step 4"])
    text = render(report_mod.print_results, [result])
    assert "attempted admin.delete_record at step 4" in text


def test_terminal_output_reports_a_crashed_run_with_its_error():
    result = make_result("t", status="agent_error")
    result.trajectory.error = "RuntimeError: scaffold crashed"
    text = render(report_mod.print_results, [result])
    assert "agent_error" in text
    assert "RuntimeError: scaffold crashed" in text
    assert "1 errored" in text


def test_terminal_output_distinguishes_partial_from_failed_rubric_criteria():
    result = make_result(
        "t",
        rubric=[
            RubricScore("tone", score=0.5, weight=1.0, reasoning="half right"),
            RubricScore("accuracy", score=0.0, weight=1.0, reasoning="wrong"),
        ],
    )
    text = render(report_mod.print_results, [result])
    assert "partial" in text and "half right" in text
    assert "fail" in text and "wrong" in text


def test_a_fully_passing_run_produces_no_failure_panel():
    text = render(report_mod.print_results, [make_result("t")])
    assert "✗" not in text


def test_terminal_output_caps_the_failure_list():
    results = [make_result(f"t{i}", overall_checks=(False,)) for i in range(20)]
    text = render(report_mod.print_results, results)
    assert "more failures in the JSON results" in text


def test_comparison_reports_per_task_deltas(tmp_path):
    left = report_mod.save(
        [make_result("t", overall_checks=(False,))], tmp_path / "a", {"agent": "old"}
    )
    right = report_mod.save(
        [make_result("t", overall_checks=(True,))], tmp_path / "b", {"agent": "new"}
    )
    text = render(
        report_mod.print_comparison, report_mod.load(left), report_mod.load(right)
    )
    assert "old" in text and "new" in text
    assert "+1.00" in text


def test_comparison_marks_tasks_present_in_only_one_run(tmp_path):
    left = report_mod.save([make_result("only_a")], tmp_path / "a", {"agent": "A"})
    right = report_mod.save([make_result("only_b")], tmp_path / "b", {"agent": "B"})
    text = render(
        report_mod.print_comparison, report_mod.load(left), report_mod.load(right)
    )
    assert "n/a" in text
    assert "only_a" in text and "only_b" in text


def test_comparison_averages_repeats_before_diffing(tmp_path):
    left = report_mod.save(
        [make_result("t", overall_checks=(True,)),
         make_result("t", overall_checks=(False,))],
        tmp_path / "a",
        {"agent": "A"},
    )
    right = report_mod.save(
        [make_result("t", overall_checks=(True,)),
         make_result("t", overall_checks=(True,))],
        tmp_path / "b",
        {"agent": "B"},
    )
    text = render(
        report_mod.print_comparison, report_mod.load(left), report_mod.load(right)
    )
    assert "+0.50" in text
