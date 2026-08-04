"""Surviving a run that does not finish, and telling a bad measurement apart
from a bad model.

Two things that only matter on long runs, and both were found the same way —
by running the harness against something real rather than reasoning about it.
"""

import asyncio
import json

import pytest

from agenteval import RunConfig, ScriptedAgent, TaskSpec, Trajectory, cli, run_suite
from agenteval import report as report_mod
from agenteval.scaffold import looks_like_an_unmade_tool_call
from agenteval.scaffold import warnings as scaffold_warnings
from agenteval.tasks import LoadedTask
from agenteval.types import Check, RubricScore, RunResult, Score, Usage


def _task(task_id="t"):
    return LoadedTask(
        spec=TaskSpec(id=task_id, prompt="p"),
        verify=lambda w, t: [Check("did it", passed=True)],
        safety=None, gold=None,
    )


def _result(task_id="t", overall_passed=True):
    return RunResult(
        task_id=task_id, agent="a", model="m",
        trajectory=Trajectory(task_id, "a"),
        score=Score(state_checks=[Check("did it", passed=overall_passed)]),
    )


# --------------------------------------------------------------------------- #
# The record round-trips
# --------------------------------------------------------------------------- #


def test_a_saved_run_rebuilds_to_the_same_record():
    """Asserted as a round trip rather than field by field. The last time this
    codebase had two separately maintained serialisations of one object, a
    field reached the file through one and vanished through the other."""
    result = RunResult(
        task_id="t", agent="a", model="m",
        trajectory=Trajectory("t", "a", model="m", final_text="done"),
        score=Score(
            state_checks=[Check("x", True, 2.0, "why")],
            rubric_scores=[RubricScore("r", 0.5, 1.0, "because")],
            safety_violations=["mailed the attacker"],
            w_state=0.9, w_rubric=0.1,
        ),
        agent_cost_usd=1.5, judge_cost_usd=0.25, judge_model="j",
        judge_usage=Usage(input_tokens=7),
        artifacts=[{"id": "f", "title": "F", "content": "c"}],
        warnings=["something looked wrong"], status="agent_error",
    )
    assert RunResult.from_dict(result.to_dict()).to_dict() == result.to_dict()


def test_the_blend_weights_survive_a_reload():
    """A resumed suite reblends every reloaded run. Without the weights in the
    record they would silently reblend at the defaults, and a run saved at
    0.9/0.1 would come back as a different number than it was printed with."""
    result = RunResult(
        task_id="t", agent="a", model=None, trajectory=Trajectory("t", "a"),
        score=Score(
            state_checks=[Check("x", True)],
            rubric_scores=[RubricScore("r", 0.0, 1.0, "")],
            w_state=0.9, w_rubric=0.1,
        ),
    )
    back = RunResult.from_dict(result.to_dict())
    assert (back.score.w_state, back.score.w_rubric) == (0.9, 0.1)
    assert back.score.overall == pytest.approx(result.score.overall)


# --------------------------------------------------------------------------- #
# The journal
# --------------------------------------------------------------------------- #


def test_a_result_is_on_disk_before_the_suite_finishes(tmp_path):
    """`results.json` is written once, at the end. That is indefensible for
    three hundred instances: a suite that dies at hour five loses every result,
    including the money already spent on them."""
    journal = report_mod.open_journal(tmp_path, {"agent": "a"})
    report_mod.append_to_journal(journal, _result("one"))
    assert [r.task_id for r in report_mod.read_journal(tmp_path)] == ["one"]


def test_a_torn_final_line_costs_one_record_not_the_run(tmp_path):
    """Exactly what a kill mid-write leaves behind. Capping the loss at the run
    in flight is the whole point of the journal."""
    journal = report_mod.open_journal(tmp_path, {"agent": "a"})
    report_mod.append_to_journal(journal, _result("one"))
    with journal.open("a") as handle:
        handle.write('{"kind": "run", "task_id": "tor')
    assert [r.task_id for r in report_mod.read_journal(tmp_path)] == ["one"]


def test_no_journal_is_not_an_error(tmp_path):
    assert report_mod.read_journal(tmp_path) == []


def test_the_journal_records_what_produced_it(tmp_path):
    report_mod.open_journal(tmp_path, {"agent": "claude:opus-5",
                                       "benchmark": {"name": "swebench:lite"}})
    header = json.loads(report_mod.journal_path(tmp_path).read_text().splitlines()[0])
    assert header["agent"] == "claude:opus-5"
    assert header["benchmark"] == "swebench:lite"


def test_opening_an_existing_journal_keeps_its_header(tmp_path):
    report_mod.open_journal(tmp_path, {"agent": "first"})
    report_mod.open_journal(tmp_path, {"agent": "first"})
    lines = report_mod.journal_path(tmp_path).read_text().splitlines()
    assert len(lines) == 1


@pytest.mark.parametrize(
    "changed", [{"agent": "ollama:qwen"}, {"benchmark": {"name": "humaneval"}}]
)
def test_resuming_somebody_elses_run_is_refused(tmp_path, changed):
    """Blending two models into one results file under one name is a mistake
    nothing downstream could detect."""
    meta = {"agent": "claude:opus-5", "benchmark": {"name": "swebench:lite"}}
    report_mod.open_journal(tmp_path, meta)
    with pytest.raises(report_mod.ResumeMismatch, match="different run"):
        report_mod.read_journal(tmp_path, {**meta, **changed})


def test_counting_what_is_already_done():
    counts = report_mod.completed_counts([_result("a"), _result("a"), _result("b")])
    assert counts == {"a": 2, "b": 1}


# --------------------------------------------------------------------------- #
# Skipping what is done
# --------------------------------------------------------------------------- #


def test_a_resumed_suite_runs_only_what_is_left():
    tasks = [_task("a"), _task("b")]
    results = asyncio.run(
        run_suite(tasks, ScriptedAgent([]), RunConfig(), completed={"a": 1})
    )
    assert [r.task_id for r in results] == ["b"]


def test_repeats_are_counted_rather_than_matched_up():
    """The fourth run of a task is not a distinct thing to pair with a saved
    one — it is one more sample of the same measurement."""
    results = asyncio.run(
        run_suite([_task("a")], ScriptedAgent([]), RunConfig(repeats=5),
                  completed={"a": 3})
    )
    assert len(results) == 2


def test_a_suite_already_finished_runs_nothing():
    results = asyncio.run(
        run_suite([_task("a")], ScriptedAgent([]), RunConfig(), completed={"a": 9})
    )
    assert results == []


def test_no_completed_map_runs_everything():
    results = asyncio.run(
        run_suite([_task("a"), _task("b")], ScriptedAgent([]), RunConfig())
    )
    assert len(results) == 2


def test_resume_end_to_end_merges_both_halves(tmp_path, monkeypatch):
    """The saved file has to contain the runs from before the crash as well as
    the ones after it, or resuming has just lost them later instead of sooner."""
    monkeypatch.setattr(cli, "credentials_available", lambda: True)
    out = tmp_path / "run"
    assert cli.main(["run", "--gold", "--out", str(out), "-c", "1"]) == 0
    before = json.loads((out / "results.json").read_text())["results"]

    # Simulate a crash after two runs: keep the header and two records.
    journal = report_mod.journal_path(out)
    kept = journal.read_text().splitlines()[:3]
    journal.write_text("\n".join(kept) + "\n")
    (out / "results.json").unlink()

    assert cli.main(["run", "--gold", "--resume", str(out), "-c", "1"]) == 0
    after = json.loads((out / "results.json").read_text())["results"]
    assert sorted(r["task_id"] for r in after) == sorted(
        r["task_id"] for r in before
    )


def test_resuming_a_mismatched_journal_exits_two(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "credentials_available", lambda: True)
    out = tmp_path / "run"
    report_mod.open_journal(out, {"agent": "someone-else",
                                  "benchmark": {"name": "local"}})
    assert cli.main(["run", "--gold", "--resume", str(out)]) == 2
    assert "Cannot resume" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Telling a bad measurement apart from a bad model
# --------------------------------------------------------------------------- #


UNMADE = json.dumps({"name": "exec_write_file",
                     "arguments": {"path": "/workspace/solution.py",
                                   "content": "def f(): pass"}})


@pytest.mark.parametrize(
    "text",
    [
        UNMADE,
        f"```json\n{UNMADE}\n```",
        "Here you go:\n\n```json\n" + UNMADE + "\n```\n",
        # Truncated mid-write, which is how it usually arrives.
        '{"name": "exec_write_file", "arguments": {"content": "def f(): pas',
    ],
)
def test_a_tool_call_written_out_as_text_is_recognised(text):
    """The observed failure: a local model returned a correct exec_write_file
    call serialised in its reply, made no call, and scored 0.00 with nothing in
    the run indicating anything had gone wrong."""
    assert looks_like_an_unmade_tool_call(text) == "exec_write_file"


@pytest.mark.parametrize(
    "text",
    [
        "I have written the solution to /workspace/solution.py.",
        "",
        '{"result": 42}',
        "```python\ndef f(): pass\n```",
    ],
)
def test_ordinary_replies_are_not_flagged(text):
    assert looks_like_an_unmade_tool_call(text) is None


def test_the_warning_says_the_score_is_not_a_measurement():
    spec = TaskSpec(id="t", prompt="p", allowed_tools=["exec_bash"])
    trajectory = Trajectory("t", "a")
    trajectory.final_text = UNMADE
    note = scaffold_warnings(spec, trajectory)[0]
    assert "exec_write_file" in note
    assert "not a capability measurement" in note


def test_a_run_that_called_nothing_at_all_is_still_flagged():
    spec = TaskSpec(id="t", prompt="p", environment={"image": "x"})
    trajectory = Trajectory("t", "a")
    trajectory.final_text = "I think the answer is 42."
    assert "nothing was measured" in scaffold_warnings(spec, trajectory)[0]


def test_an_empty_response_is_the_clearest_case_of_all():
    """An earlier version of this required some text to complain about, and so
    let through the one case with no ambiguity in it. Observed on 8 of 20
    HumanEval runs: one turn, end_turn, no content, no calls, status ok, scored
    0.00. There is no reading of that as a capability result."""
    spec = TaskSpec(id="t", prompt="p", environment={"image": "x"})
    trajectory = Trajectory("t", "a")  # no final_text, no messages
    note = scaffold_warnings(spec, trajectory)[0]
    assert "empty response" in note
    assert "failed request, not a score" in note


def test_a_run_that_used_its_tools_is_not_flagged():
    from agenteval.types import ToolCall

    spec = TaskSpec(id="t", prompt="p", allowed_tools=["exec_bash"])
    trajectory = Trajectory("t", "a")
    trajectory.calls = [ToolCall(1, "exec_bash", {}, "ok")]
    trajectory.final_text = UNMADE  # even if it also wrote one out
    assert scaffold_warnings(spec, trajectory) == []


def test_a_task_that_needs_no_tools_may_answer_in_prose():
    """Not every task is a tool-use task, and a prose answer to a prose
    question is not a broken scaffold."""
    spec = TaskSpec(id="t", prompt="p")
    trajectory = Trajectory("t", "a")
    trajectory.final_text = "The answer is 42."
    assert scaffold_warnings(spec, trajectory) == []


def test_warnings_reach_the_saved_run():
    result = _result()
    result.warnings = ["the scaffold never engaged"]
    assert RunResult.from_dict(result.to_dict()).warnings == result.warnings


def test_warnings_are_printed_above_the_failures(capsys):
    from rich.console import Console

    result = _result(overall_passed=False)
    result.warnings = ["the model wrote a call out as text instead of calling it"]
    report_mod.print_results([result], Console())
    out = capsys.readouterr().out
    assert "may not be measuring anything" in out
    assert "wrote a call out as text" in out


def test_a_clean_suite_says_nothing_about_scaffolding(capsys):
    from rich.console import Console

    report_mod.print_results([_result()], Console())
    assert "may not be measuring" not in capsys.readouterr().out
