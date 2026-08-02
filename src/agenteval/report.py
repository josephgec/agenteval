"""Turning results into something a human can act on.

Three surfaces: a terminal table for the run you just did, a JSON file that is
the durable record, and a standalone HTML report for sharing. The terminal view
leads with failed checks, because "0.62" is not actionable and "the escalation
went to the wrong manager" is.
"""

from __future__ import annotations

import html
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .types import RunResult


def _esc(value: object) -> str:
    """Escape anything interpolated into the HTML report.

    Most of these strings originate outside the harness. The judge's `reasoning`
    is the sharpest case: it is instructed to quote spans from the artifacts, so
    it carries agent-authored text verbatim into a page meant to be shared.
    """
    return html.escape(str(value), quote=True)


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


def save(results: list[RunResult], out_dir: Path, meta: dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
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
    path = out_dir / "results.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "results.json"
    return json.loads(path.read_text())


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

_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>agenteval — {agent}</title>
<style>
:root {{ color-scheme: light dark; --fg:#111; --bg:#fff; --dim:#666; --line:#e4e4e7; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --fg:#e8e8ea; --bg:#0f0f11; --dim:#9a9aa2; --line:#2a2a30; }}
}}
body {{ font: 15px/1.55 ui-sans-serif,-apple-system,Segoe UI,sans-serif;
  margin:0; padding:2.5rem 1.25rem; color:var(--fg); background:var(--bg); }}
main {{ max-width: 62rem; margin: 0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 .25rem; }}
.sub {{ color:var(--dim); margin-bottom:2rem; }}
table {{ width:100%; border-collapse:collapse; margin-bottom:2.5rem; }}
th,td {{ text-align:left; padding:.5rem .6rem; border-bottom:1px solid var(--line); }}
th {{ font-weight:600; font-size:.8rem; text-transform:uppercase;
  letter-spacing:.04em; color:var(--dim); }}
td.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
.bar {{ display:inline-block; height:.5rem; border-radius:99px; min-width:2px;
  vertical-align:middle; }}
.pass {{ background:#16a34a; }} .mid {{ background:#d97706; }} .fail {{ background:#dc2626; }}
.card {{ border:1px solid var(--line); border-radius:.6rem; padding:1rem 1.1rem;
  margin-bottom:1rem; }}
.card h3 {{ margin:0 0 .5rem; font-size:1rem; }}
li.x {{ color:#dc2626; }} li.p {{ color:#d97706; }}
ul {{ margin:.35rem 0 0; padding-left:1.2rem; }}
.detail {{ color:var(--dim); }}
.wrap {{ overflow-x:auto; }}
</style></head><body><main>
<h1>agenteval</h1>
<div class="sub">{agent} · {runs} runs · mean {mean:.3f} · ${cost:.3f} · {saved}</div>
<div class="wrap"><table>
<tr><th>Task</th><th>Overall</th><th></th><th>State</th><th>Rubric</th>
<th>Safe</th><th>Steps</th><th>Cost</th></tr>
{rows}
</table></div>
<h2 style="font-size:1.1rem">Findings</h2>
{findings}
</main></body></html>
"""


def write_html(results: list[RunResult], out_dir: Path, meta: dict[str, Any]) -> Path:
    stats = aggregate(results)
    rows = []
    for task_id, s in stats.items():
        mean = s["overall_mean"]
        cls = "pass" if mean >= 0.85 else "mid" if mean >= 0.5 else "fail"
        rows.append(
            f"<tr><td>{_esc(task_id)}</td><td class='n'>{mean:.2f}</td>"
            f"<td><span class='bar {cls}' style='width:{max(2, mean * 80):.0f}px'>"
            "</span></td>"
            f"<td class='n'>{s['state_mean']:.2f}</td>"
            f"<td class='n'>"
            f"{'—' if s['rubric_mean'] is None else format(s['rubric_mean'], '.2f')}"
            "</td>"
            f"<td class='n'>{'yes' if not s['unsafe_runs'] else 'NO'}</td>"
            f"<td class='n'>{s['steps_mean']:.0f}</td>"
            f"<td class='n'>${s['cost_total']:.3f}</td></tr>"
        )

    findings = []
    for result in results:
        failed = [c for c in result.score.state_checks if not c.passed]
        weak = [r for r in result.score.rubric_scores if r.score < 1.0]
        if not failed and not weak and result.score.safe:
            continue
        items = [
            f"<li class='x'>unsafe: {_esc(v)}</li>"
            for v in result.score.safety_violations
        ]
        items += [
            f"<li class='x'>{_esc(c.name)} "
            f"<span class='detail'>— {_esc(c.detail)}</span></li>"
            for c in failed
        ]
        items += [
            f"<li class='p'>{_esc(r.id)} "
            f"<span class='detail'>— {_esc(r.reasoning)}</span></li>"
            for r in weak
        ]
        findings.append(
            f"<div class='card'><h3>{_esc(result.task_id)}</h3>"
            f"<ul>{''.join(items)}</ul></div>"
        )

    page = _HTML_TEMPLATE.format(
        agent=_esc(meta.get("agent", "unknown")),
        runs=len(results),
        mean=statistics.fmean([r.score.overall for r in results]) if results else 0.0,
        cost=sum(r.cost_usd for r in results),
        saved=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        rows="\n".join(rows),
        findings="\n".join(findings) or "<p class='detail'>No findings.</p>",
    )
    path = Path(out_dir) / "report.html"
    path.write_text(page)
    return path
