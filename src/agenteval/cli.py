"""Command line entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import webbrowser
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import benchmarks as bench
from . import design
from . import sandbox as sandbox_mod
from . import report as report_mod
from . import ui
from .agents import ScriptedAgent, build_agent
from .grading.judge import LLMJudge
from .runner import RunConfig, run_suite
from .tasks import DEFAULT_TASK_ROOT, LoadedTask, TaskError, filter_by_tag

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


def make_sandbox(args: argparse.Namespace):
    """The sandbox, if asked for, checked before anything depends on it."""
    if not getattr(args, "sandbox", False):
        return None
    box = sandbox_mod.Sandbox(timeout=args.sandbox_timeout)
    box.preflight()
    return box


def gather(args: argparse.Namespace) -> tuple[list[LoadedTask], "bench.Benchmark"]:
    """Resolve the benchmark and load the instances asked for.

    Everything that runs tasks goes through here, including the local task
    directory. Keeping one path means the downloaded benchmarks are exercised
    by the same code that has been running the hand-written suite all along,
    rather than sitting on a parallel track that only gets tested when someone
    remembers to.
    """
    # `--tasks` is the local benchmark's argument, so the flag keeps working
    # unchanged and there is no special case for "the default benchmark".
    benchmark = bench.resolve(args.benchmark or f"local:{args.tasks}")

    box = make_sandbox(args)
    if box is not None:
        # Verifier isolation only means something where the verifier is
        # untrusted Python from a task directory. A downloaded benchmark's
        # adapter is code in this repository; pretending --sandbox covered it
        # would be a false assurance, so say so instead.
        if not hasattr(benchmark, "sandbox"):
            raise TaskError(
                f"--sandbox does not apply to the {benchmark.name} benchmark: it "
                "isolates task-supplied verify.py, and this benchmark is graded "
                "by adapter code in agenteval itself."
            )
        benchmark.sandbox = box

    tasks = bench.load_tasks(
        benchmark,
        only=getattr(args, "task", None) or None,
        limit=getattr(args, "limit", None),
        seed=getattr(args, "seed", 0),
    )
    if getattr(args, "tag", None):
        tasks = filter_by_tag(tasks, args.tag)
    return tasks, benchmark


def report_images_to_pull(tasks: list[LoadedTask]) -> None:
    """Say what a run is about to download, before it downloads it.

    A benchmark with per-instance images is a different proposition from one
    with a shared image: `agenteval --benchmark swebench run` across SWE-bench
    Lite is three hundred pulls of about a gigabyte each. Finding that out from
    a full disk two hours in is a bad way to find it out, and the fix is one
    line of arithmetic before anything starts.
    """
    from .exec import environment as env_mod

    images = {
        t.spec.environment["image"]
        for t in tasks
        if t.spec.environment and t.spec.environment.get("image")
    }
    if len(images) < 2 or not env_mod.available():
        return
    missing = [i for i in sorted(images) if not env_mod.image_present(i)]
    if not missing:
        return
    console.print(
        f"[yellow]{len(missing)} of {len(images)} container images are not "
        f"local[/yellow] and will be pulled on demand. For benchmarks that "
        f"publish one image per instance that is roughly a gigabyte each — "
        f"about {len(missing)} GB here. [dim]Use --limit to sample.[/dim]"
    )


def cmd_benchmarks(args: argparse.Namespace) -> int:
    table = Table(title="benchmarks", header_style="bold")
    table.add_column("name")
    table.add_column("what it is")
    for name, description in sorted(bench.registered().items()):
        # Escaped: a description mentioning `pip install 'agenteval[swebench]'`
        # is otherwise rendered as `agenteval`, Rich having read the extras as
        # a style tag — an install line that silently drops the extra.
        table.add_row(name, escape(description))
    console.print()
    console.print(table)
    console.print(
        "[dim]select with --benchmark; anything after a colon is the "
        "benchmark's own argument (a directory, a split, an image)[/dim]"
    )
    return 0


def cmd_sandbox(args: argparse.Namespace) -> int:
    box = sandbox_mod.Sandbox()
    if args.action == "build":
        console.print(f"[dim]building {box.image}…[/dim]")
        box.build()
        console.print(f"[green]built[/green] {box.image}")
        return 0
    box.preflight()
    console.print(f"[green]ready[/green] {box.image}")
    console.print(
        "[dim]task code runs with no network, no environment, a read-only "
        "root and all capabilities dropped[/dim]"
    )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    tasks, benchmark = gather(args)
    table = Table(title=f"{len(tasks)} tasks in {benchmark.name}", header_style="bold")
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
    tasks, benchmark = gather(args)
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
    report_images_to_pull(tasks)

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
            # Each task replays its own reference trajectory, so the agent
            # depends on the task — the one case run_suite takes a factory for.
            agent = lambda task: ScriptedAgent(task.gold or [], name="gold")  # noqa: E731
        else:
            # build_agent drops options the chosen backend does not understand,
            # so --max-turns behaves the same whichever one is running.
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
        # A benchmark score means nothing without the subset it came from.
        # "20 of HumanEval's 164" is a different claim from "HumanEval", and
        # the difference has to survive into the saved results or the number
        # gets quoted as the second one.
        "benchmark": bench.summarise(benchmark) | {"ran": len(tasks)},
    }
    json_path = report_mod.save(results, out_dir, meta, tasks)
    html_path = report_mod.write_html(results, out_dir, meta, tasks)

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


def cmd_show(args: argparse.Namespace) -> int:
    payload = report_mod.load(Path(args.run))
    try:
        result = report_mod.select_run(payload, args.task, args.index)
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        return 2
    if not args.task:
        console.print(
            f"[dim]showing the lowest-scoring run "
            f"({result['task_id']}); pass --task to pick another[/dim]"
        )
    report_mod.print_trajectory(result, console, full=args.full)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Rebuild a run's report and open it.

    Regeneration matters because the report is written once, at run time. An
    improvement to the explorer would otherwise only reach runs you pay to
    repeat — this rebuilds from the saved results instead.
    """
    run = Path(args.run)
    payload = report_mod.load(run)
    directory = run if run.is_dir() else run.parent
    path = ui.write(payload, directory)
    console.print(f"[dim]{path}[/dim]")
    if not args.no_open:
        webbrowser.open(path.resolve().as_uri())
    return 0


def cmd_design(args: argparse.Namespace) -> int:
    """Rebuild the design-system specimens from the live stylesheet.

    Composed from the same TOKENS and COMPONENTS the report renders with, so a
    change to either shows up here on the next build rather than leaving the
    library quietly describing an older design.
    """
    written = design.build(args.out)
    for path in written:
        console.print(f"[dim]{path}[/dim]")
    console.print(
        f"{len(written)} specimens in [bold]{args.out}[/bold] · "
        "push with the DesignSync tool to sync them to claude.ai/design"
    )
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
    parser.add_argument(
        "--benchmark",
        help="Where tasks come from (default: the local ./tasks directory). "
        "Anything after a colon is that benchmark's own argument, e.g. "
        "humaneval, local:/path/to/tasks. See `agenteval benchmarks`.",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Run task code (verify.py) in a Docker container with no network "
        "and no environment. Use for task suites you did not write.",
    )
    parser.add_argument(
        "--sandbox-timeout", type=float, default=60.0,
        help="Wall-clock ceiling per sandboxed call (default 60s)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sandbox = sub.add_parser(
        "sandbox", help="Build or check the task-sandbox container image"
    )
    p_sandbox.add_argument("action", choices=["build", "check"])
    p_sandbox.set_defaults(func=cmd_sandbox)

    p_benchmarks = sub.add_parser(
        "benchmarks", help="List the benchmarks that can supply tasks"
    )
    p_benchmarks.set_defaults(func=cmd_benchmarks)

    p_list = sub.add_parser("list", help="List available tasks")
    p_list.add_argument("--task", action="append", help="Only this instance")
    p_list.add_argument("--tag", help="Only tasks carrying this tag")
    p_list.add_argument(
        "--limit", type=int, help="Sample this many instances (see `run --limit`)"
    )
    p_list.add_argument("--seed", type=int, default=0)
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
        "--limit",
        type=int,
        help="Run a random sample of this many instances. Random rather than "
        "the first N: benchmarks are usually ordered by something, so a prefix "
        "is a biased sample that still looks like a whole-benchmark score.",
    )
    p_run.add_argument(
        "--seed", type=int, default=0, help="Seed for --limit (default 0)"
    )
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

    p_show = sub.add_parser(
        "show", help="Print one run's trajectory alongside how it was graded"
    )
    p_show.add_argument("run", help="Run directory or results.json")
    p_show.add_argument(
        "--task", help="Which task (default: the lowest-scoring run)"
    )
    p_show.add_argument(
        "--index", type=int, default=0,
        help="Which repeat of that task, 0-based (default 0)",
    )
    p_show.add_argument(
        "--full", action="store_true",
        help="Show thinking blocks and untruncated tool arguments and results",
    )
    p_show.set_defaults(func=cmd_show)

    p_ui = sub.add_parser(
        "ui", help="Rebuild a run's HTML report and open it in a browser"
    )
    p_ui.add_argument("run", help="Run directory or results.json")
    p_ui.add_argument(
        "--no-open", action="store_true", help="Write the file without opening it"
    )
    p_ui.set_defaults(func=cmd_ui)

    p_design = sub.add_parser(
        "design", help="Rebuild the design-system specimens from the stylesheet"
    )
    p_design.add_argument("--out", default="design", help="Output directory")
    p_design.set_defaults(func=cmd_design)

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
    except bench.BenchmarkError as exc:
        console.print(f"[red]Benchmark error:[/red] {exc}")
        return 2
    except sandbox_mod.SandboxError as exc:
        console.print(f"[red]Sandbox error:[/red] {exc}")
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
