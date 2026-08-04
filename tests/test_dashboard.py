"""Every run in one place.

The interesting logic is arithmetic over saved runs, so almost everything here
builds small results.json files on disk and checks what comes back. The two
things worth being strict about: that `gold` is never mistaken for a model, and
that a task which has stopped informing is visible as such.
"""

import json

import pytest

from agenteval import cli, dashboard


def _run(tmp_path, name, agent, benchmark, scores, saved_at, cost=0.0,
         warnings=None, status="ok", safe=True):
    """Write a minimal results.json the dashboard can read."""
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    results = [
        {
            "task_id": task, "agent": agent, "model": None, "status": status,
            "score": {"overall": score, "state": score, "rubric": None,
                      "safe": safe, "safety_violations": [], "checks": [],
                      "rubric_scores": []},
            "process": {"steps": 1, "turns": 1, "wall_seconds": 1.0,
                        "cost_usd": 0.0},
            "warnings": list(warnings or []),
        }
        for task, score in scores.items()
    ]
    payload = {
        "tasks": {},
        "meta": {
            "agent": agent, "saved_at": saved_at, "runs": len(results),
            "mean_overall": sum(scores.values()) / len(scores) if scores else 0.0,
            "total_cost_usd": cost,
            "benchmark": {"name": benchmark, "instances": 100,
                          "ran": len(results)},
        },
        "results": results,
    }
    (directory / "results.json").write_text(json.dumps(payload))
    return directory


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def test_runs_are_read_newest_first(tmp_path):
    _run(tmp_path, "old", "a", "local", {"t": 1.0}, "2026-01-01T00:00:00")
    _run(tmp_path, "new", "a", "local", {"t": 1.0}, "2026-06-01T00:00:00")
    assert [r.path for r in dashboard.collect(tmp_path)] == ["new", "old"]


def test_a_malformed_run_is_skipped_not_fatal(tmp_path):
    """Half the value of this page is looking at a collection that includes an
    interrupted run. Refusing to render because one file is broken would be
    exactly backwards."""
    _run(tmp_path, "good", "a", "local", {"t": 1.0}, "2026-01-01T00:00:00")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "results.json").write_text("{ not json")
    assert [r.path for r in dashboard.collect(tmp_path)] == ["good"]


def test_an_empty_directory_renders_rather_than_raising(tmp_path):
    payload = dashboard.build(tmp_path)
    assert payload["totals"]["runs"] == 0
    assert "No runs found" in dashboard.render(payload)


def test_runs_from_before_the_benchmark_layer_say_local(tmp_path):
    """Older saved runs have no benchmark block; naming the source beats an
    empty cell."""
    directory = tmp_path / "old"
    directory.mkdir()
    (directory / "results.json").write_text(json.dumps({
        "meta": {"agent": "a", "saved_at": "2026-01-01T00:00:00", "runs": 1,
                 "mean_overall": 1.0, "total_cost_usd": 0.0},
        "results": [],
    }))
    assert dashboard.collect(tmp_path)[0].benchmark == "local"


# --------------------------------------------------------------------------- #
# Gold is not a model
# --------------------------------------------------------------------------- #


def test_gold_does_not_count_toward_spread(tmp_path):
    """It scores 1.00 by construction. Counting it makes every task look
    discriminating, which is the opposite of the truth."""
    _run(tmp_path, "g", "gold", "local", {"t": 1.0}, "2026-01-01T00:00:00")
    _run(tmp_path, "m", "model-a", "local", {"t": 1.0}, "2026-01-02T00:00:00")
    row = dashboard.by_task(dashboard.collect(tmp_path))[0]
    assert row["spread"] is None      # one model, not two
    assert row["headroom"] == 0.0     # and that model already solves it
    assert row["solvable"] == 1.0     # gold still shown, as its own column


def test_gold_is_kept_out_of_the_agent_columns(tmp_path):
    _run(tmp_path, "g", "gold", "local", {"t": 1.0}, "2026-01-01T00:00:00")
    _run(tmp_path, "m", "model-a", "local", {"t": 0.5}, "2026-01-02T00:00:00")
    assert dashboard.build(tmp_path)["agents"] == ["model-a"]


def test_a_task_only_gold_has_run_is_marked_untested(tmp_path):
    """Solvable, and nothing else known. A row of dashes reads like a dud."""
    _run(tmp_path, "g", "gold", "local", {"t": 1.0}, "2026-01-01T00:00:00")
    row = dashboard.by_task(dashboard.collect(tmp_path))[0]
    assert row["headroom"] is None and row["spread"] is None
    assert "no model has run it" in dashboard.render(dashboard.build(tmp_path))


# --------------------------------------------------------------------------- #
# Does this task tell you anything
# --------------------------------------------------------------------------- #


def test_spread_is_best_model_minus_worst(tmp_path):
    _run(tmp_path, "a", "model-a", "local", {"t": 0.9}, "2026-01-01T00:00:00")
    _run(tmp_path, "b", "model-b", "local", {"t": 0.2}, "2026-01-02T00:00:00")
    assert dashboard.by_task(dashboard.collect(tmp_path))[0]["spread"] == (
        pytest.approx(0.7)
    )


def test_headroom_catches_saturation_with_only_one_model(tmp_path):
    """The measure that matters before you have a second frontier model. A task
    the best model already solves cannot rank anything above it."""
    _run(tmp_path, "a", "model-a", "local",
         {"solved": 1.0, "hard": 0.3}, "2026-01-01T00:00:00")
    rows = {r["task"]: r for r in dashboard.by_task(dashboard.collect(tmp_path))}
    assert rows["solved"]["headroom"] == pytest.approx(0.0)
    assert rows["hard"]["headroom"] == pytest.approx(0.7)


def test_uninformative_tasks_sort_first(tmp_path):
    """They are the finding, so they should not be buried at the bottom of a
    long table."""
    _run(tmp_path, "a", "model-a", "local",
         {"hard": 0.1, "solved": 1.0}, "2026-01-01T00:00:00")
    assert [r["task"] for r in dashboard.by_task(dashboard.collect(tmp_path))] == [
        "solved", "hard"
    ]


def test_repeats_of_a_task_average_before_comparing(tmp_path):
    _run(tmp_path, "a", "model-a", "local", {"t": 1.0}, "2026-01-01T00:00:00")
    _run(tmp_path, "b", "model-a", "local", {"t": 0.0}, "2026-01-02T00:00:00")
    row = dashboard.by_task(dashboard.collect(tmp_path))[0]
    assert row["agents"]["model-a"] == pytest.approx(0.5)
    assert row["n"] == 2


# --------------------------------------------------------------------------- #
# The leaderboard
# --------------------------------------------------------------------------- #


def test_rows_are_one_agent_and_one_benchmark(tmp_path):
    _run(tmp_path, "a", "m", "local", {"t": 1.0}, "2026-01-01T00:00:00")
    _run(tmp_path, "b", "m", "humaneval", {"t": 0.5}, "2026-01-02T00:00:00")
    rows = dashboard.leaderboard(dashboard.collect(tmp_path))
    assert {(r["agent"], r["benchmark"]) for r in rows} == {
        ("m", "local"), ("m", "humaneval")
    }


def test_the_latest_run_is_reported_beside_the_average(tmp_path):
    """They diverge whenever a task or a verifier changed underneath, which
    happens — one run in this project's own history scored 0.50 on a task a bug
    was zeroing, and averaging it with the fixed run describes neither."""
    _run(tmp_path, "before", "m", "local", {"t": 0.0}, "2026-01-01T00:00:00")
    _run(tmp_path, "after", "m", "local", {"t": 1.0}, "2026-06-01T00:00:00")
    row = dashboard.leaderboard(dashboard.collect(tmp_path))[0]
    assert row["mean"] == pytest.approx(0.5)
    assert row["latest_mean"] == pytest.approx(1.0)
    assert row["runs"] == 2


def test_the_best_agent_is_first(tmp_path):
    _run(tmp_path, "a", "weak", "local", {"t": 0.1}, "2026-01-01T00:00:00")
    _run(tmp_path, "b", "strong", "local", {"t": 0.9}, "2026-01-02T00:00:00")
    assert dashboard.leaderboard(dashboard.collect(tmp_path))[0]["agent"] == "strong"


def test_trust_signals_are_counted_per_row(tmp_path):
    """A model that never emitted a tool call otherwise sits at the bottom of a
    leaderboard looking merely bad."""
    _run(tmp_path, "a", "m", "local", {"t": 0.0}, "2026-01-01T00:00:00",
         warnings=["the scaffold never engaged"], status="agent_error",
         safe=False)
    row = dashboard.leaderboard(dashboard.collect(tmp_path))[0]
    assert row["warnings"] == 1 and row["errors"] == 1 and row["unsafe"] == 1


def test_totals_add_up(tmp_path):
    _run(tmp_path, "a", "m", "local", {"x": 1.0, "y": 0.0},
         "2026-01-01T00:00:00", cost=1.25)
    _run(tmp_path, "b", "n", "humaneval", {"z": 1.0}, "2026-01-02T00:00:00",
         cost=0.75)
    totals = dashboard.build(tmp_path)["totals"]
    assert totals == {"runs": 2, "measurements": 3, "cost": 2.0, "agents": 2,
                      "benchmarks": 2, "warnings": 0}


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #


def test_the_payload_cannot_close_the_script_block(tmp_path):
    """Task ids and agent names reach the page verbatim; a literal
    `</script>` in one would close the block early."""
    _run(tmp_path, "a", "</script><b>x", "local", {"t": 1.0},
         "2026-01-01T00:00:00")
    html = dashboard.render(dashboard.build(tmp_path))
    assert "</script><b>x" not in html
    assert "\\u003c/script>" in html


def test_writing_puts_the_page_beside_the_runs(tmp_path):
    """So the per-run links resolve as plain relative paths and the whole thing
    works from file://.

    The href is assembled in the page from the run's directory name, so what is
    checkable here is that the name reaches the payload and that the template
    around it is relative — no leading slash, no absolute path from this
    machine that would break the moment the directory moved.
    """
    _run(tmp_path, "somerun", "m", "local", {"t": 1.0}, "2026-01-01T00:00:00")
    path = dashboard.write(tmp_path)
    assert path == tmp_path / "index.html"
    page = path.read_text()
    assert '"path": "somerun"' in page
    assert 'href="${esc(r.path)}/report.html"' in page
    assert str(tmp_path) not in page


def test_an_explicit_output_path_is_honoured(tmp_path):
    _run(tmp_path, "a", "m", "local", {"t": 1.0}, "2026-01-01T00:00:00")
    out = tmp_path / "elsewhere" / "page.html"
    assert dashboard.write(tmp_path, out) == out and out.exists()


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #


def test_dashboard_reports_saturation_in_the_terminal(tmp_path, capsys):
    _run(tmp_path, "a", "m", "local", {"solved": 1.0, "hard": 0.2},
         "2026-01-01T00:00:00")
    assert cli.main(["dashboard", "--runs", str(tmp_path), "--no-open"]) == 0
    out = capsys.readouterr().out
    assert "1 of 2" in out and "already solved by the best model" in out


def test_dashboard_flags_suspect_runs(tmp_path, capsys):
    _run(tmp_path, "a", "m", "local", {"t": 0.0}, "2026-01-01T00:00:00",
         warnings=["wrote a call out as text"])
    cli.main(["dashboard", "--runs", str(tmp_path), "--no-open"])
    assert "may not be measuring anything" in capsys.readouterr().out


def test_dashboard_on_an_empty_directory_says_so(tmp_path, capsys):
    assert cli.main(["dashboard", "--runs", str(tmp_path), "--no-open"]) == 1
    assert "No runs found" in capsys.readouterr().out
