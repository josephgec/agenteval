"""CLI surface: agent specs, argument validation, exit codes, and outputs.

Exit codes are the contract CI depends on, so they are asserted directly:
    0  ran, and cleared --fail-under if one was given
    1  ran, but scored below --fail-under
    2  refused to run (bad flags, missing credentials, unknown task)
"""

import json
from pathlib import Path

import pytest

from agenteval import ClaudeAgent, build_agent, cli

#: Captured before the autouse fixture can replace it, so the credential
#: detection tests below exercise the real implementation rather than the stub.
REAL_CREDENTIALS_AVAILABLE = cli.credentials_available


@pytest.fixture
def tasks_root():
    from agenteval.tasks import DEFAULT_TASK_ROOT

    return str(DEFAULT_TASK_ROOT)


@pytest.fixture(autouse=True)
def no_ambient_credentials(monkeypatch):
    """Force the no-credential path so results do not depend on whether the
    developer running the suite happens to be logged in."""
    monkeypatch.setattr(cli, "credentials_available", lambda: False)


# --------------------------------------------------------------------------- #
# Agent specs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec,model,effort",
    [
        ("claude", "claude-opus-5", "high"),
        ("claude:sonnet-5", "claude-sonnet-5", "high"),
        ("claude:opus-5:medium", "claude-opus-5", "medium"),
        ("claude:claude-opus-4-8", "claude-opus-4-8", "high"),
        ("claude:claude-haiku-4-5:low", "claude-haiku-4-5", "low"),
    ],
)
def test_agent_spec_parsing(spec, model, effort):
    agent = build_agent(spec)
    assert isinstance(agent, ClaudeAgent)
    assert agent.model == model
    assert agent.effort == effort


def test_unknown_agent_kind_is_rejected_with_a_useful_message():
    with pytest.raises(ValueError, match="unknown agent 'gpt'"):
        build_agent("gpt:4")


def test_overrides_reach_the_agent():
    agent = build_agent("claude:opus-5", max_turns=3, thinking=False)
    assert agent.max_turns == 3 and agent.thinking is False


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #


def test_disabled_thinking_above_high_effort_is_refused_before_any_api_call(
    tasks_root, capsys
):
    """The API rejects this pairing; failing here costs nothing."""
    code = cli.main(
        ["--tasks", tasks_root, "run", "--task", "ticket_triage",
         "--no-thinking", "--effort", "max"]
    )
    assert code == 2
    assert "cannot be combined" in capsys.readouterr().out


def test_that_validation_runs_even_without_credentials(tasks_root, capsys):
    """A bad flag combination should be reported on any machine, not masked by
    the credential check."""
    code = cli.main(
        ["--tasks", tasks_root, "run", "--task", "ticket_triage",
         "--no-thinking", "--effort", "xhigh"]
    )
    assert code == 2
    assert "credentials" not in capsys.readouterr().out.lower()


def test_missing_credentials_refuse_the_run_and_point_at_gold(tasks_root, capsys):
    code = cli.main(["--tasks", tasks_root, "run", "--task", "ticket_triage"])
    assert code == 2
    out = capsys.readouterr().out
    assert "ant auth login" in out and "--gold" in out


def test_an_unknown_task_id_is_a_clean_error(tasks_root, capsys):
    code = cli.main(["--tasks", tasks_root, "run", "--gold", "--task", "nope"])
    assert code == 2
    assert "unknown task" in capsys.readouterr().out.lower()


def test_a_tag_that_matches_nothing_is_reported(tasks_root, capsys):
    code = cli.main(
        ["--tasks", tasks_root, "run", "--gold", "--tag", "does-not-exist"]
    )
    assert code == 1
    assert "No tasks matched" in capsys.readouterr().out


def test_gold_refuses_tasks_without_a_reference_solution(tmp_path, capsys):
    task_dir = tmp_path / "bare"
    task_dir.mkdir()
    (task_dir / "task.yaml").write_text("id: bare\nprompt: do it\n")
    (task_dir / "verify.py").write_text("def verify(world, trajectory):\n    return []\n")

    code = cli.main(["--tasks", str(tmp_path), "run", "--gold"])
    assert code == 1
    assert "No GOLD trajectory" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Gold runs end to end
# --------------------------------------------------------------------------- #


def test_gold_run_succeeds_and_writes_both_artifacts(tasks_root, tmp_path, capsys):
    out = tmp_path / "run"
    code = cli.main(
        ["--tasks", tasks_root, "run", "--gold", "--task", "ticket_triage",
         "--out", str(out)]
    )
    assert code == 0
    assert (out / "results.json").exists()
    assert (out / "report.html").exists()

    payload = json.loads((out / "results.json").read_text())
    assert payload["meta"]["agent"] == "gold"
    assert payload["results"][0]["score"]["overall"] == 1.0
    # No judge is constructed for a gold run, so nothing is billed.
    assert payload["meta"]["judge_model"] is None
    assert payload["meta"]["total_cost_usd"] == 0.0


def test_fail_under_gates_on_the_mean(tasks_root, tmp_path):
    args = ["--tasks", tasks_root, "run", "--gold", "--task", "ticket_triage",
            "--out", str(tmp_path / "r")]
    assert cli.main([*args, "--fail-under", "0.99"]) == 0
    assert cli.main([*args, "--fail-under", "1.01"]) == 1


def test_repeats_produce_one_result_per_run(tasks_root, tmp_path):
    out = tmp_path / "run"
    cli.main(
        ["--tasks", tasks_root, "run", "--gold", "--task", "ticket_triage",
         "-k", "3", "--out", str(out)]
    )
    payload = json.loads((out / "results.json").read_text())
    assert len(payload["results"]) == 3
    assert payload["meta"]["repeats"] == 3


def test_default_output_directory_is_timestamped_and_slugged():
    path = cli._default_out_dir("claude:claude-opus-5:high")
    assert path.parent == Path("runs")
    # ':' is legal on POSIX but breaks on Windows and in shell completion.
    assert ":" not in path.name
    assert path.name.endswith("-claude-claude-opus-5-high")


# --------------------------------------------------------------------------- #
# The real run path
# --------------------------------------------------------------------------- #


@pytest.fixture
def stubbed_run(monkeypatch):
    """Exercise the non-gold path — agent construction, judge wiring, save,
    report — without credentials or API calls."""
    from agenteval import ScriptedAgent
    from agenteval.types import RubricScore

    built = {}

    def fake_build_agent(spec, **overrides):
        agent = ScriptedAgent(
            [
                {"tool": "admin.set_policy_override",
                 "input": {"policy_doc_id": "policy/expenses", "reason": "r"}},
                {"say": "done"},
            ],
            name=spec,
        )
        built.update({"spec": spec, **overrides})
        return agent

    class StubJudge:
        def __init__(self, model, effort):
            built["judge_model"] = model
            built["judge_effort"] = effort
            self.model = model

        async def score(self, task, world, trajectory):
            from agenteval.grading.judge import JudgeOutcome
            from agenteval.types import Usage

            built["judged"] = True
            return JudgeOutcome(
                scores=[RubricScore("tone", score=0.5, weight=1.0,
                                    reasoning="ok")],
                usage=Usage(input_tokens=8000, output_tokens=400),
                model=self.model,
            )

    monkeypatch.setattr(cli, "credentials_available", lambda: True)
    monkeypatch.setattr(cli, "build_agent", fake_build_agent)
    monkeypatch.setattr(cli, "LLMJudge", StubJudge)
    return built


def test_run_wires_agent_and_judge_from_the_flags(stubbed_run, tasks_root, tmp_path):
    out = tmp_path / "run"
    code = cli.main(
        ["--tasks", tasks_root, "run", "--agent", "claude:sonnet-5:medium",
         "--task", "expense_approval", "--max-turns", "7",
         "--judge-model", "claude-opus-5", "--judge-effort", "low",
         "--out", str(out)]
    )
    assert code == 0
    assert stubbed_run["spec"] == "claude:sonnet-5:medium"
    assert stubbed_run["max_turns"] == 7
    assert stubbed_run["judge_model"] == "claude-opus-5"
    assert stubbed_run["judge_effort"] == "low"
    assert stubbed_run["judged"] is True

    payload = json.loads((out / "results.json").read_text())
    assert payload["meta"]["judge_model"] == "claude-opus-5"
    # The agent is scripted and free; the judge is not. Reporting the run as
    # costless was the accounting bug this guards against.
    assert payload["results"][0]["process"]["agent_cost_usd"] == 0.0
    assert payload["results"][0]["process"]["judge_cost_usd"] > 0
    assert payload["meta"]["judge_cost_usd"] > 0
    # The scripted agent reaches for a forbidden tool, so the run is unsafe and
    # the overall score is gated to zero regardless of the rubric.
    assert payload["results"][0]["score"]["safe"] is False
    assert payload["results"][0]["score"]["overall"] == 0.0
    assert payload["results"][0]["score"]["rubric"] == 0.5


def test_no_judge_skips_judging_entirely(stubbed_run, tasks_root, tmp_path):
    out = tmp_path / "run"
    cli.main(
        ["--tasks", tasks_root, "run", "--task", "expense_approval",
         "--no-judge", "--out", str(out)]
    )
    assert "judged" not in stubbed_run
    payload = json.loads((out / "results.json").read_text())
    assert payload["meta"]["judge_model"] is None
    assert payload["results"][0]["score"]["rubric"] is None


def test_unsafe_runs_are_marked_in_the_progress_line(
    stubbed_run, tasks_root, tmp_path, capsys
):
    cli.main(
        ["--tasks", tasks_root, "run", "--task", "expense_approval",
         "--no-judge", "--out", str(tmp_path / "run")]
    )
    assert "!" in capsys.readouterr().out


def test_custom_score_weights_are_recorded(stubbed_run, tasks_root, tmp_path):
    out = tmp_path / "run"
    cli.main(
        ["--tasks", tasks_root, "run", "--task", "expense_approval",
         "--w-state", "0.9", "--w-rubric", "0.1", "--out", str(out)]
    )
    meta = json.loads((out / "results.json").read_text())["meta"]
    assert meta["weights"] == {"state": 0.9, "rubric": 0.1}


def test_a_broken_tasks_directory_is_a_clean_error(capsys):
    code = cli.main(["--tasks", "/nonexistent/path", "list"])
    assert code == 2
    assert "Task error" in capsys.readouterr().out


def test_interrupting_a_run_exits_130_without_a_traceback(
    monkeypatch, tasks_root, capsys
):
    """Ctrl-C during a long suite is routine, not a crash."""
    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "discover", interrupt)
    assert cli.main(["--tasks", tasks_root, "run", "--gold"]) == 130
    assert "Interrupted" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Reporting commands
# --------------------------------------------------------------------------- #


def test_list_names_every_task(tasks_root, capsys, monkeypatch):
    from agenteval.tasks import discover

    # Rich sizes tables to the terminal and elides overflowing cells. Pin a
    # wide one so this asserts on content rather than on terminal geometry.
    monkeypatch.setenv("COLUMNS", "200")

    assert cli.main(["--tasks", tasks_root, "list"]) == 0
    out = capsys.readouterr().out
    for task in discover(tasks_root):
        assert task.id in out


def test_report_reads_back_a_saved_run(tasks_root, tmp_path, capsys):
    out = tmp_path / "run"
    cli.main(["--tasks", tasks_root, "run", "--gold", "--task", "ticket_triage",
              "--out", str(out)])
    capsys.readouterr()

    assert cli.main(["report", str(out)]) == 0
    text = capsys.readouterr().out
    assert "ticket_triage" in text and "gold" in text


def test_compare_diffs_two_saved_runs(tasks_root, tmp_path, capsys):
    for name in ("a", "b"):
        cli.main(["--tasks", tasks_root, "run", "--gold", "--task", "ticket_triage",
                  "--out", str(tmp_path / name)])
    capsys.readouterr()

    assert cli.main(["compare", str(tmp_path / "a"), str(tmp_path / "b")]) == 0
    assert "Comparison" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Credential detection
# --------------------------------------------------------------------------- #


def test_an_api_key_counts_as_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert REAL_CREDENTIALS_AVAILABLE() is True


def test_an_auth_token_counts_as_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    assert REAL_CREDENTIALS_AVAILABLE() is True


def test_an_oauth_profile_counts_even_with_no_env_var(monkeypatch):
    """An unset ANTHROPIC_API_KEY does not mean there are no credentials — the
    SDK also resolves an `ant auth login` profile, so the CLI must check for
    one before telling anybody to go and find a key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *a, **k: cli.subprocess.CompletedProcess(a, returncode=0),
    )
    assert REAL_CREDENTIALS_AVAILABLE() is True


def test_a_logged_out_profile_does_not_count(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *a, **k: cli.subprocess.CompletedProcess(a, returncode=1),
    )
    assert REAL_CREDENTIALS_AVAILABLE() is False


@pytest.mark.parametrize("failure", [FileNotFoundError, TimeoutError])
def test_probing_for_the_ant_binary_never_raises(monkeypatch, failure):
    """A machine without the CLI installed should get the friendly message,
    not a traceback."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    def boom(*a, **k):
        raise (
            cli.subprocess.TimeoutExpired("ant", 10)
            if failure is TimeoutError
            else FileNotFoundError
        )

    monkeypatch.setattr(cli.subprocess, "run", boom)
    assert REAL_CREDENTIALS_AVAILABLE() is False
