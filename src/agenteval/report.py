"""Turning results into something a human can act on.

Three surfaces: a terminal table for the run you just did, a JSON file that is
the durable record, and a standalone HTML report for sharing. The terminal view
leads with failed checks, because "0.62" is not actionable and "the escalation
went to the wrong manager" is.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import ui
from .types import RunResult


def _grade_color(value: float) -> str:
    if value >= 0.85:
        return "green"
    if value >= 0.5:
        return "yellow"
    return "red"


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def aggregate(results: list[RunResult]) -> dict[str, dict[str, Any]]:
    """Collapse repeats into per-task statistics."""
    by_task: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        by_task[r.task_id].append(r)

    out: dict[str, dict[str, Any]] = {}
    for task_id, runs in sorted(by_task.items()):
        overalls = [r.score.overall for r in runs]
        out[task_id] = {
            "n": len(runs),
            "overall_mean": statistics.fmean(overalls),
            # With repeats, spread matters as much as the mean: an agent that
            # scores 1.0 then 0.2 is not a "0.6 agent", it is an unreliable one.
            "overall_stdev": (
                statistics.stdev(overalls) if len(overalls) > 1 else 0.0
            ),
            "state_mean": statistics.fmean(
                [r.score.state_score for r in runs if r.score.state_score is not None]
                or [0.0]
            ),
            "rubric_mean": (
                statistics.fmean(
                    [
                        r.score.rubric_score
                        for r in runs
                        if r.score.rubric_score is not None
                    ]
                )
                if any(r.score.rubric_score is not None for r in runs)
                else None
            ),
            "unsafe_runs": sum(1 for r in runs if not r.score.safe),
            "errors": sum(1 for r in runs if r.status != "ok"),
            "steps_mean": statistics.fmean([r.trajectory.steps for r in runs]),
            "cost_total": sum(r.cost_usd for r in runs),
            "agent_cost_total": sum(r.agent_cost_usd for r in runs),
            "judge_cost_total": sum(r.judge_cost_usd for r in runs),
            "seconds_mean": statistics.fmean(
                [r.trajectory.wall_seconds for r in runs]
            ),
            "runs": runs,
        }
    return out


# --------------------------------------------------------------------------- #
# Terminal
# --------------------------------------------------------------------------- #


def print_results(results: list[RunResult], console: Console | None = None) -> None:
    console = console or Console()
    stats = aggregate(results)

    table = Table(title="Results by task", header_style="bold", expand=False)
    table.add_column("task")
    table.add_column("n", justify="right")
    table.add_column("overall", justify="right")
    table.add_column("±", justify="right")
    table.add_column("state", justify="right")
    table.add_column("rubric", justify="right")
    table.add_column("safe", justify="center")
    table.add_column("steps", justify="right")
    table.add_column("sec", justify="right")
    table.add_column("cost", justify="right")

    for task_id, s in stats.items():
        mean = s["overall_mean"]
        safe_cell = (
            "[green]ok[/green]"
            if not s["unsafe_runs"]
            else f"[red]{s['unsafe_runs']}/{s['n']}[/red]"
        )
        table.add_row(
            task_id,
            str(s["n"]),
            f"[{_grade_color(mean)}]{mean:.2f}[/{_grade_color(mean)}]",
            f"{s['overall_stdev']:.2f}" if s["n"] > 1 else "—",
            f"{s['state_mean']:.2f}",
            "—" if s["rubric_mean"] is None else f"{s['rubric_mean']:.2f}",
            safe_cell,
            f"{s['steps_mean']:.0f}",
            f"{s['seconds_mean']:.0f}",
            f"${s['cost_total']:.3f}",
        )

    console.print()
    console.print(table)

    overall = statistics.fmean([r.score.overall for r in results]) if results else 0.0
    total_cost = sum(r.cost_usd for r in results)
    judge_cost = sum(r.judge_cost_usd for r in results)
    unsafe = sum(1 for r in results if not r.score.safe)
    errored = sum(1 for r in results if r.status != "ok")
    summary = (
        f"[bold]{len(results)}[/bold] runs   "
        f"mean [bold {_grade_color(overall)}]{overall:.3f}[/bold {_grade_color(overall)}]   "
        f"cost [bold]${total_cost:.3f}[/bold]"
    )
    if judge_cost:
        # Broken out because it is easy to forget the judge is a second billed
        # model — and with a local agent it is the entire bill.
        summary += (
            f" [dim](agent ${total_cost - judge_cost:.3f} + "
            f"judge ${judge_cost:.3f})[/dim]"
        )
    if unsafe:
        summary += f"   [red]{unsafe} unsafe[/red]"
    if errored:
        summary += f"   [yellow]{errored} errored[/yellow]"
    console.print(Panel(summary, expand=False, border_style="dim"))

    # Printed above the failures, not among them. A suite where the scaffold
    # never engaged is not a suite of low scores — it is a suite of numbers
    # that measured nothing, and reading it as the former is the whole reason
    # this exists.
    suspect = [r for r in results if r.warnings]
    if suspect:
        console.print(
            f"[yellow]{len(suspect)} of {len(results)} runs may not be "
            f"measuring anything:[/yellow]"
        )
        for note in sorted({w for r in suspect for w in r.warnings}):
            console.print(f"  [yellow]·[/yellow] {note}")

    _print_failures(results, console)


def _print_failures(results: list[RunResult], console: Console) -> None:
    """The part people actually read: what went wrong and where."""
    shown = 0
    for result in results:
        failed = [c for c in result.score.state_checks if not c.passed]
        weak = [r for r in result.score.rubric_scores if r.score < 1.0]
        if not failed and not weak and result.score.safe and result.status == "ok":
            continue
        if shown >= 12:
            console.print("[dim]…more failures in the JSON results.[/dim]")
            break
        shown += 1

        lines: list[str] = []
        if result.status != "ok":
            lines.append(f"[yellow]status[/yellow] {result.status}: "
                         f"{result.trajectory.error}")
        for violation in result.score.safety_violations:
            lines.append(f"[red]unsafe[/red] {violation}")
        for check in failed:
            lines.append(f"[red]✗[/red] {check.name} — [dim]{check.detail}[/dim]")
        for rubric in weak:
            label = "partial" if rubric.score > 0 else "fail"
            lines.append(
                f"[yellow]{label}[/yellow] {rubric.id} — [dim]{rubric.reasoning}[/dim]"
            )
        console.print(
            Panel(
                "\n".join(lines),
                title=f"{result.task_id}  [dim]({result.agent})[/dim]",
                title_align="left",
                border_style="red" if failed or not result.score.safe else "yellow",
            )
        )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def build_payload(
    results: list[RunResult],
    meta: dict[str, Any],
    tasks: "list[Any] | None" = None,
) -> dict[str, Any]:
    """The saved shape, built once and shared by the JSON file and the report.

    Assembling it twice is how the two drift apart, and the HTML is generated
    from exactly what lands on disk.

    `tasks` carries each task's definition — prompt, seeded world, rubric and
    source files — so a saved result stays interpretable after the task changes
    underneath it.
    """
    return {
        "tasks": {t.id: t.manifest() for t in (tasks or [])},
        "meta": {
            **meta,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "runs": len(results),
            "mean_overall": (
                statistics.fmean([r.score.overall for r in results])
                if results
                else 0.0
            ),
            "total_cost_usd": sum(r.cost_usd for r in results),
            "agent_cost_usd": sum(r.agent_cost_usd for r in results),
            "judge_cost_usd": sum(r.judge_cost_usd for r in results),
        },
        "results": [r.to_dict() for r in results],
    }


def save(
    results: list[RunResult],
    out_dir: Path,
    meta: dict[str, Any],
    tasks: "list[Any] | None" = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(json.dumps(build_payload(results, meta, tasks), indent=2))
    return path


def load(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "results.json"
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Trajectory inspection
# --------------------------------------------------------------------------- #


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def print_trajectory(
    result: dict[str, Any], console: Console | None = None, full: bool = False
) -> None:
    """Render one run so a failure can be read rather than reconstructed.

    Interleaves what the agent did with how it was graded, because the question
    being asked is almost always "which of these calls was the wrong one", and
    answering it from a score and a JSON blob is needlessly hard.
    """
    console = console or Console()
    score, process = result["score"], result["process"]
    limit = 10_000 if full else 160

    status_colour = "green" if result["status"] == "ok" else "yellow"
    console.print(
        f"\n[bold]{result['task_id']}[/bold] · {result['agent']} · "
        f"[{status_colour}]{result['status']}[/{status_colour}]"
    )
    console.print(
        f"overall [bold {_grade_color(score['overall'])}]{score['overall']:.2f}"
        f"[/bold {_grade_color(score['overall'])}]   "
        f"state {'—' if score['state'] is None else format(score['state'], '.2f')}   "
        f"rubric {'—' if score['rubric'] is None else format(score['rubric'], '.2f')}   "
        f"{'[green]safe[/green]' if score['safe'] else '[red]UNSAFE[/red]'}   "
        f"[dim]{process['steps']} steps · {process['turns']} turns · "
        f"{process['wall_seconds']:.0f}s · ${process['cost_usd']:.3f}[/dim]"
    )
    if process.get("error"):
        console.print(f"[yellow]error[/yellow] {process['error']}")

    trajectory = result["trajectory"]
    if trajectory["thinking"] and full:
        console.print("\n[bold dim]thinking[/bold dim]")
        for block in trajectory["thinking"]:
            console.print(f"  [dim]{_clip(block, limit)}[/dim]")
    elif trajectory["thinking"]:
        console.print(
            f"\n[dim]{len(trajectory['thinking'])} thinking block(s) "
            "— pass --full to show[/dim]"
        )

    console.print("\n[bold dim]tool calls[/bold dim]")
    if not trajectory["calls"]:
        console.print("  [dim](none)[/dim]")
    for call in trajectory["calls"]:
        if call["blocked_reason"]:
            marker = f"[red]✗[/red] [dim]{call['blocked_reason']}[/dim]"
        elif call["is_error"]:
            marker = "[yellow]![/yellow]"
        else:
            marker = "[green]·[/green]"
        console.print(
            f"  {marker} [bold]{call['step']:>3}[/bold] {call['name']}  "
            f"[dim]{_clip(call['input'], limit)}[/dim]"
        )
        console.print(f"      [dim]→ {_clip(call['output'], limit)}[/dim]")

    if trajectory["final_text"]:
        console.print("\n[bold dim]final message[/bold dim]")
        console.print(f"  {_clip(trajectory['final_text'], limit)}")

    console.print("\n[bold dim]checks[/bold dim]")
    for check in score["checks"]:
        mark = "[green]✓[/green]" if check["passed"] else "[red]✗[/red]"
        detail = f" [dim]— {check['detail']}[/dim]" if not check["passed"] else ""
        console.print(f"  {mark} {check['name']}{detail}")

    if score["rubric_scores"]:
        console.print("\n[bold dim]rubric[/bold dim]")
        for item in score["rubric_scores"]:
            colour = _grade_color(item["score"])
            console.print(
                f"  [{colour}]{item['score']:.1f}[/{colour}] {item['id']} "
                f"[dim]— {_clip(item['reasoning'], limit)}[/dim]"
            )

    if score["safety_violations"]:
        console.print("\n[bold red]safety[/bold red]")
        for violation in score["safety_violations"]:
            console.print(f"  [red]![/red] {violation}")
    console.print()


def select_run(
    payload: dict[str, Any], task_id: str | None = None, index: int = 0
) -> dict[str, Any]:
    """Pick which run to render.

    With no task named, defaults to the lowest-scoring one — that is virtually
    always the run you opened the file to look at.
    """
    results = payload["results"]
    if task_id:
        matching = [r for r in results if r["task_id"] == task_id]
        if not matching:
            available = sorted({r["task_id"] for r in results})
            raise KeyError(
                f"no run for task {task_id!r} in this result set; "
                f"available: {', '.join(available)}"
            )
        if index >= len(matching):
            raise KeyError(
                f"task {task_id!r} has {len(matching)} run(s), no index {index}"
            )
        return matching[index]
    if not results:
        raise KeyError("this result set is empty")
    return min(results, key=lambda r: r["score"]["overall"])


# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #


def print_comparison(
    left: dict[str, Any],
    right: dict[str, Any],
    console: Console | None = None,
) -> None:
    console = console or Console()

    def by_task(payload: dict[str, Any]) -> dict[str, list[float]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for r in payload["results"]:
            grouped[r["task_id"]].append(r["score"]["overall"])
        return grouped

    left_scores, right_scores = by_task(left), by_task(right)
    left_label = left["meta"].get("agent", "A")
    right_label = right["meta"].get("agent", "B")

    table = Table(title="Comparison", header_style="bold")
    table.add_column("task")
    table.add_column(left_label, justify="right")
    table.add_column(right_label, justify="right")
    table.add_column("delta", justify="right")

    for task_id in sorted(set(left_scores) | set(right_scores)):
        a = statistics.fmean(left_scores[task_id]) if task_id in left_scores else None
        b = statistics.fmean(right_scores[task_id]) if task_id in right_scores else None
        if a is None or b is None:
            delta_cell = "[dim]n/a[/dim]"
        else:
            delta = b - a
            color = "green" if delta > 0.01 else "red" if delta < -0.01 else "dim"
            delta_cell = f"[{color}]{delta:+.2f}[/{color}]"
        table.add_row(
            task_id,
            "—" if a is None else f"{a:.2f}",
            "—" if b is None else f"{b:.2f}",
            delta_cell,
        )

    console.print()
    console.print(table)
    console.print(
        f"[dim]{left_label}: ${left['meta']['total_cost_usd']:.3f}   "
        f"{right_label}: ${right['meta']['total_cost_usd']:.3f}[/dim]"
    )


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #


def write_html(
    results: list[RunResult],
    out_dir: Path,
    meta: dict[str, Any],
    tasks: "list[Any] | None" = None,
) -> Path:
    """Write the standalone run explorer. See `agenteval.ui` for the design."""
    return ui.write(build_payload(results, meta, tasks), out_dir)


# --------------------------------------------------------------------------- #
# The journal: surviving a run that does not finish
# --------------------------------------------------------------------------- #
#
# `results.json` is written once, at the end. That is fine for five simulated
# tasks and indefensible for three hundred SWE-bench instances: a suite that
# dies at hour five loses every result, including the money already spent on
# them. The journal is the same records appended as they land, one JSON object
# per line, so a crash costs the run in flight and nothing else.

JOURNAL = "journal.jsonl"


class ResumeMismatch(Exception):
    """The journal was written by a different run than the one resuming it."""


def journal_path(out_dir: Path) -> Path:
    return Path(out_dir) / JOURNAL


def open_journal(out_dir: Path, meta: dict[str, Any]) -> Path:
    """Start (or continue) a journal, recording what produced it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = journal_path(out_dir)
    if not path.exists():
        header = {"kind": "header", "agent": meta.get("agent"),
                  "benchmark": (meta.get("benchmark") or {}).get("name"),
                  "repeats": meta.get("repeats")}
        path.write_text(json.dumps(header) + "\n")
    return path


def append_to_journal(path: Path, result: RunResult) -> None:
    # Opened and closed per record rather than held: the whole point is to
    # survive a process that does not get to run its cleanup.
    with Path(path).open("a") as handle:
        handle.write(json.dumps({"kind": "run", **result.to_dict()}) + "\n")
        handle.flush()


def read_journal(out_dir: Path, meta: dict[str, Any] | None = None) -> list[RunResult]:
    """Completed runs from a previous attempt, if any.

    A truncated final line is dropped rather than fatal — it is exactly what a
    kill mid-write leaves behind, and losing one record is the cost the journal
    exists to cap.
    """
    path = journal_path(out_dir)
    if not path.exists():
        return []
    results: list[RunResult] = []
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("kind") == "header" and meta is not None:
            _check_resumable(entry, meta)
        elif entry.get("kind") == "run":
            results.append(RunResult.from_dict(entry))
    return results


def _check_resumable(header: dict[str, Any], meta: dict[str, Any]) -> None:
    """Refuse to continue somebody else's run.

    Resuming a Claude suite with a local agent would blend two models into one
    results file under one name, and nothing downstream could tell.
    """
    now = {"agent": meta.get("agent"),
           "benchmark": (meta.get("benchmark") or {}).get("name")}
    was = {"agent": header.get("agent"), "benchmark": header.get("benchmark")}
    differences = [k for k in now if was.get(k) is not None and was[k] != now[k]]
    if differences:
        detail = ", ".join(f"{k}: {was[k]!r} -> {now[k]!r}" for k in differences)
        raise ResumeMismatch(
            f"this journal was written by a different run ({detail}). Resuming "
            "would blend them into one results file under one name."
        )


def completed_counts(results: list[RunResult]) -> dict[str, int]:
    """How many runs of each task are already done."""
    counts: dict[str, int] = {}
    for result in results:
        counts[result.task_id] = counts.get(result.task_id, 0) + 1
    return counts
