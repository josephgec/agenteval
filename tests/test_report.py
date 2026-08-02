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


def test_agent_authored_text_cannot_escape_the_embedded_payload(tmp_path):
    """The report embeds the run verbatim and renders it in the browser, so the
    containment question moved: agent text now sits inside a <script> block and
    a literal `</script>` in an email body would end it early and execute the
    remainder. The judge quotes spans from the artifacts, so this is
    model-authored text arriving in a page meant to be shared.
    """
    result = make_result("t", overall_checks=(False,), steps=1)
    result.trajectory.calls[0].input = {
        "body": "</script><script>alert(1)</script>"
    }
    result.score.rubric_scores = [
        RubricScore("tone", score=0.0, weight=1.0,
                    reasoning="</SCRIPT ><img src=x onerror=alert(2)>")
    ]
    result.score.safety_violations = ["</script>"]

    page = report_mod.write_html([result], tmp_path, {"agent": "a"}).read_text()
    embedded = page.split('id="payload">')[1].split("</script>")[0]

    # The precise property: nothing in the payload can open a tag, so it cannot
    # close the block or introduce an element. The characters survive as data —
    # `onerror=alert(2)` is still present and still inert, which is why
    # substring-hunting would test the wrong thing.
    assert "<" not in embedded
    assert page.count("</script>") == page.count("<script")

    # And it round-trips, so the reader still sees exactly what the agent wrote.
    payload = json.loads(embedded)
    assert payload["results"][0]["score"]["safety_violations"] == ["</script>"]
    assert (
        payload["results"][0]["trajectory"]["calls"][0]["input"]["body"]
        == "</script><script>alert(1)</script>"
    )


def test_a_dangerous_agent_label_cannot_break_the_header(tmp_path):
    """`agent` is interpolated into markup rather than into the payload."""
    page = report_mod.write_html(
        [make_result()], tmp_path, {"agent": '<img src=x onerror=alert(1)>'}
    ).read_text()
    assert "<img" not in page  # no element introduced, in the title or the header
    assert page.count("&lt;img src=x onerror=alert(1)&gt;") == 2


def test_html_report_is_self_contained(tmp_path):
    """It has to render identically offline, from file://, years later. Any
    external reference is a dependency on something outliving the report."""
    page = report_mod.write_html([make_result()], tmp_path, {"agent": "a"}).read_text()
    for external in ("http://", "https://", "src=", "@import", "//fonts."):
        assert external not in page


def test_the_report_embeds_exactly_what_was_saved(tmp_path):
    """The HTML is generated from the same payload as results.json, so the two
    can never disagree about what happened."""
    results = [make_result("t", overall_checks=(False,), cost=0.25)]
    meta = {"agent": "claude:opus-5"}
    saved = json.loads(report_mod.save(results, tmp_path / "r", meta).read_text())
    page = report_mod.write_html(results, tmp_path / "r", meta).read_text()

    embedded = json.loads(page.split('id="payload">')[1].split("</script>")[0])
    assert embedded["results"] == saved["results"]
    assert embedded["meta"]["agent"] == saved["meta"]["agent"]


# --------------------------------------------------------------------------- #
# Terminal and comparison
# --------------------------------------------------------------------------- #


def render(fn, *args) -> str:
    console = Console(record=True, width=200, force_terminal=False)
    fn(*args, console)
    return console.export_text()


def render_full(result, full: bool = False) -> str:
    console = Console(record=True, width=200, force_terminal=False)
    report_mod.print_trajectory(result.to_dict(), console, full=full)
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


# --------------------------------------------------------------------------- #
# Trajectory inspection
# --------------------------------------------------------------------------- #


def payload_with(*results):
    return {"meta": {"agent": "a"}, "results": [r.to_dict() for r in results]}


def test_select_run_defaults_to_the_worst_run():
    """Almost always the one you opened the file to look at."""
    payload = payload_with(
        make_result("good", overall_checks=(True,)),
        make_result("bad", overall_checks=(False,)),
        make_result("mixed", overall_checks=(True, False)),
    )
    assert report_mod.select_run(payload)["task_id"] == "bad"


def test_select_run_picks_a_named_task_and_repeat():
    payload = payload_with(
        make_result("t", overall_checks=(True,)),
        make_result("t", overall_checks=(False,)),
        make_result("other"),
    )
    assert report_mod.select_run(payload, "t", 0)["score"]["overall"] == 1.0
    assert report_mod.select_run(payload, "t", 1)["score"]["overall"] == 0.0


def test_select_run_names_what_is_available_when_the_task_is_unknown():
    payload = payload_with(make_result("alpha"), make_result("beta"))
    with pytest.raises(KeyError, match="alpha, beta"):
        report_mod.select_run(payload, "gamma")


def test_select_run_rejects_an_out_of_range_repeat():
    payload = payload_with(make_result("t"))
    with pytest.raises(KeyError, match="1 run"):
        report_mod.select_run(payload, "t", 5)


def test_select_run_on_an_empty_result_set():
    with pytest.raises(KeyError, match="empty"):
        report_mod.select_run({"meta": {}, "results": []})


def test_trajectory_view_interleaves_actions_with_grading():
    """The question being answered is 'which call was the wrong one', so the
    calls and the checks have to appear together."""
    result = make_result("expense_approval", overall_checks=(False,), steps=2)
    result.score.state_checks[0].name = "EXP-502 escalated to direct manager"
    result.score.state_checks[0].detail = "expected 'EMP-003', got None"
    result.score.safety_violations = ["attempted forbidden tool admin_delete"]
    result.score.rubric_scores = [
        RubricScore("tone", score=0.5, weight=1.0, reasoning="curt")
    ]
    text = render_full(result)

    assert "expense_approval" in text
    assert "tickets_get" in text                        # the calls
    assert "EXP-502 escalated to direct manager" in text  # the checks
    assert "expected 'EMP-003', got None" in text         # why it failed
    assert "attempted forbidden tool admin_delete" in text
    assert "tone" in text and "curt" in text
    assert "UNSAFE" in text


def test_trajectory_view_truncates_by_default_and_expands_with_full():
    result = make_result("t", steps=1)
    result.trajectory.calls[0].output = "x" * 4000
    result.trajectory.thinking = ["a long private deliberation"]

    clipped = render_full(result, full=False)
    assert "…" in clipped
    assert "thinking block(s)" in clipped          # summarised, not shown
    assert "a long private deliberation" not in clipped

    expanded = render_full(result, full=True)
    assert "a long private deliberation" in expanded
    assert expanded.count("x") > 3000


def test_trajectory_view_marks_blocked_and_errored_calls():
    result = make_result("t", steps=2)
    result.trajectory.calls[0].blocked_reason = "forbidden"
    result.trajectory.calls[1].is_error = True
    text = render_full(result)
    assert "forbidden" in text


def test_trajectory_view_handles_a_run_that_did_nothing():
    result = make_result("t", steps=0, status="agent_error")
    result.trajectory.error = "RuntimeError: crashed"
    text = render_full(result)
    assert "(none)" in text
    assert "RuntimeError: crashed" in text
    assert "agent_error" in text


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
