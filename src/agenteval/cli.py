"""Command line entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import report as report_mod
from .agents import ScriptedAgent, build_agent
from .grading.judge import LLMJudge
from .runner import RunConfig, run_suite
from .tasks import DEFAULT_TASK_ROOT, TaskError, discover, filter_by_tag

console = Console()


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #


def credentials_available() -> bool:
    """An unset ANTHROPIC_API_KEY does not mean there are no credentials.

    The SDK also resolves an `ant auth login` profile, so check that before
    telling anyone to go find a key.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    try:
        done = subprocess.run(
            ["ant", "auth", "status"], capture_output=True, timeout=10
        )
        return done.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def report_missing_credentials(judge_only: bool = False) -> None:
    console.print(
        "[red]No Anthropic credentials found.[/red]\n"
        "Either run [bold]ant auth login[/bold] (stores a profile the SDK reads "
        "automatically), or export [bold]ANTHROPIC_API_KEY[/bold]."
    )
    if judge_only:
        # The agent is local; only rubric grading needs the API.
        console.print(
            "The agent itself is local — only the LLM judge needs the API. "
            "Add [bold]--no-judge[/bold] to run on state assertions alone."
        )
    else:
        console.print(
            "To exercise the harness with no credentials and no cost, use "
            "[bold]agenteval run --gold[/bold]."
        )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_list(args: argparse.Namespace) -> int:
    tasks = discover(args.tasks)
    table = Table(title=f"{len(tasks)} tasks in {args.tasks}", header_style="bold")
    table.add_column("id")
    table.add_column("tags")
    table.add_column("tools", justify="right")
    table.add_column("forbidden")
    table.add_column("rubric", justify="right")
    table.add_column("gold", justify="center")
    table.add_column("max steps", justify="right")
    for task in tasks:
        spec = task.spec
        table.add_row(
            spec.id,
            ", ".join(spec.tags) or "—",
            str(len(spec.allowed_tools) or "all"),
            ", ".join(spec.forbidden_tools) or "—",
            str(len(spec.rubric) or "—"),
            "[green]yes[/green]" if task.gold else "[dim]no[/dim]",
            str(spec.max_steps),
        )
    console.print()
    console.print(table)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    tasks = discover(args.tasks, only=args.task or None)
    if args.tag:
        tasks = filter_by_tag(tasks, args.tag)
    if not tasks:
        console.print("[red]No tasks matched.[/red]")
        return 1

    if args.gold:
        missing = [t.id for t in tasks if not t.gold]
        if missing:
            console.print(
                f"[red]No GOLD trajectory defined for: {', '.join(missing)}[/red]"
            )
            return 1
        judge = None
        agent_label = "gold"
    else:
        backend = args.agent.partition(":")[0]

        # Disabling thinking is only legal at effort `high` or below; pairing it
        # with xhigh/max is a 400. Checked before credentials so a bad flag
        # combination is reported even on a machine with no key configured.
        if backend == "claude" and not args.thinking and args.effort in (
            "xhigh", "max"
        ):
            console.print(
                f"[red]--no-thinking cannot be combined with --effort {args.effort}"
                "[/red] (the API rejects disabled thinking above `high` effort). "
                "Use --effort high or lower, or drop --no-thinking."
            )
            return 2

        # A local agent needs no Anthropic credentials — but the judge always
        # runs on Claude, so rubric grading still does.
        judging = not args.no_judge
        if (backend == "claude" or judging) and not credentials_available():
            report_missing_credentials(judge_only=backend != "claude")
            return 2

        judge = (
            None
            if args.no_judge
            else LLMJudge(model=args.judge_model, effort=args.judge_effort)
        )
        agent_label = args.agent

    config = RunConfig(
        repeats=args.repeats,
        concurrency=args.concurrency,
        judge=judge,
        w_state=args.w_state,
        w_rubric=args.w_rubric,
    )

    total = len(tasks) * config.repeats
    console.print(
        f"\n[bold]{agent_label}[/bold] · {len(tasks)} tasks × {config.repeats} "
        f"= {total} runs · concurrency {config.concurrency}"
        + ("" if judge else " · [dim]judge off[/dim]")
    )

    done = 0

    def progress(task_id: str, result) -> None:
        nonlocal done
        if result is None:
            return
        done += 1
        mark = (
            "[green]●[/green]"
            if result.score.overall >= 0.85
            else "[yellow]●[/yellow]"
            if result.score.overall >= 0.5
            else "[red]●[/red]"
        )
        if not result.score.safe:
            mark = "[red]![/red]"
        console.print(
            f"  {mark} [{done:>3}/{total}] {task_id:<22} "
            f"{result.score.overall:.2f}  "
            f"[dim]{result.trajectory.steps} steps, "
            f"{result.trajectory.wall_seconds:.0f}s, "
            f"${result.cost_usd:.3f}[/dim]"
        )

    async def go():
        if args.gold:
            results = []
            for task in tasks:
                agent = ScriptedAgent(task.gold or [], name="gold")
                for _ in range(config.repeats):
                    from .runner import run_one

                    result = await run_one(task, agent, config)
                    progress(task.id, result)
                    results.append(result)
            return results
        # build_agent drops options the chosen backend does not understand, so
        # --max-turns works the same way regardless of which one is running.
        agent = build_agent(
            args.agent,
            max_turns=args.max_turns,
            thinking=args.thinking,
            num_ctx=args.num_ctx,
            host=args.ollama_host,
        )
        return await run_suite(tasks, agent, config, on_progress=progress)

    results = asyncio.run(go())

    out_dir = Path(args.out) if args.out else _default_out_dir(agent_label)
    meta = {
        "agent": agent_label,
        "judge_model": None if not judge else judge.model,
        "repeats": config.repeats,
        "tasks": [t.id for t in tasks],
        "weights": {"state": config.w_state, "rubric": config.w_rubric},
    }
    json_path = report_mod.save(results, out_dir, meta)
    html_path = report_mod.write_html(results, out_dir, meta)

    report_mod.print_results(results, console)
    console.print(f"[dim]results  {json_path}[/dim]")
    console.print(f"[dim]report   {html_path}[/dim]")

    if args.fail_under is not None:
        import statistics

        mean = statistics.fmean([r.score.overall for r in results])
        if mean < args.fail_under:
            console.print(
                f"[red]mean {mean:.3f} is below --fail-under "
                f"{args.fail_under:.3f}[/red]"
            )
            return 1
    return 0


def _default_out_dir(agent_label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", agent_label).strip("-")
    return Path("runs") / f"{stamp}-{slug}"


def cmd_report(args: argparse.Namespace) -> int:
    payload = report_mod.load(Path(args.run))
    meta = payload["meta"]
    console.print(
        f"\n[bold]{meta.get('agent')}[/bold] · {meta['runs']} runs · "
        f"mean {meta['mean_overall']:.3f} · ${meta['total_cost_usd']:.3f} · "
        f"{meta.get('saved_at', '')}"
    )
    table = Table(header_style="bold")
    table.add_column("task")
    table.add_column("overall", justify="right")
    table.add_column("state", justify="right")
    table.add_column("rubric", justify="right")
    table.add_column("safe", justify="center")
    table.add_column("steps", justify="right")
    table.add_column("cost", justify="right")
    for r in payload["results"]:
        s, p = r["score"], r["process"]
        table.add_row(
            r["task_id"],
            f"{s['overall']:.2f}",
            "—" if s["state"] is None else f"{s['state']:.2f}",
            "—" if s["rubric"] is None else f"{s['rubric']:.2f}",
            "[green]ok[/green]" if s["safe"] else "[red]NO[/red]",
            str(p["steps"]),
            f"${p['cost_usd']:.3f}",
        )
    console.print(table)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    report_mod.print_comparison(
        report_mod.load(Path(args.left)), report_mod.load(Path(args.right)), console
    )
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agenteval",
        description="Evaluate agents on simulated enterprise workflows.",
    )
    parser.add_argument(
        "--tasks",
        default=str(DEFAULT_TASK_ROOT),
        help="Directory of task definitions (default: ./tasks)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List available tasks")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="Run a suite and grade it")
    p_run.add_argument(
        "--agent",
        default="claude",
        help="Agent spec: claude | claude:<model> | claude:<model>:<effort> | "
        "ollama:<model tag>, e.g. ollama:qwen2.5:7b-instruct",
    )
    p_run.add_argument("--task", action="append", help="Run only this task (repeatable)")
    p_run.add_argument("--tag", help="Run only tasks carrying this tag")
    p_run.add_argument(
        "-k", "--repeats", type=int, default=1, help="Runs per task (variance)"
    )
    p_run.add_argument("-c", "--concurrency", type=int, default=4)
    p_run.add_argument("--max-turns", type=int, default=20)
    p_run.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Disable extended thinking (only valid at effort high or below)",
    )
    p_run.add_argument(
        "--effort",
        default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Only used to validate the --no-thinking pairing; set effort in "
        "--agent as claude:<model>:<effort>",
    )
    p_run.add_argument(
        "--num-ctx",
        type=int,
        default=32768,
        help="Ollama context window. Ollama truncates silently past this, so "
        "raising it is usually cheaper than debugging the result (default 32768)",
    )
    p_run.add_argument(
        "--ollama-host", default="http://localhost:11434", help="Ollama base URL"
    )
    p_run.add_argument("--no-judge", action="store_true", help="Skip rubric grading")
    p_run.add_argument("--judge-model", default="claude-opus-5")
    p_run.add_argument(
        "--judge-effort", default="high",
        choices=["low", "medium", "high", "xhigh", "max"]
    )
    p_run.add_argument("--w-state", type=float, default=0.7)
    p_run.add_argument("--w-rubric", type=float, default=0.3)
    p_run.add_argument("--out", help="Output directory (default: runs/<timestamp>)")
    p_run.add_argument(
        "--gold",
        action="store_true",
        help="Replay each task's reference solution instead of calling a model. "
        "Free, offline, and validates that the tasks are solvable.",
    )
    p_run.add_argument(
        "--fail-under",
        type=float,
        help="Exit non-zero if the mean score falls below this (for CI)",
    )
    p_run.set_defaults(func=cmd_run, thinking=True)

    p_report = sub.add_parser("report", help="Print a saved run")
    p_report.add_argument("run", help="Run directory or results.json")
    p_report.set_defaults(func=cmd_report)

    p_compare = sub.add_parser("compare", help="Diff two saved runs")
    p_compare.add_argument("left")
    p_compare.add_argument("right")
    p_compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except TaskError as exc:
        console.print(f"[red]Task error:[/red] {exc}")
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
