"""Every run, in one place.

`ui.py` explains one run. This explains a directory of them — which model is
ahead, whether anything regressed, what it all cost, and which tasks are
carrying any information at all.

That last one is the reason this exists rather than being a scoreboard. A task
every model solves has stopped measuring anything; it costs money on every run
and moves no number. The enterprise suite here is already in that state for
frontier models, and the only way to see it is to put the runs side by side and
look at the spread per task. So the dashboard leads with discrimination, not
with a leaderboard.

Two rules carried over from everything else in this project.

*A score without its provenance is a misleading score.* A twenty-instance
sample of HumanEval and a full run of it must not sit in a table looking like
the same claim, so the subset is on the face of every row.

*A run that measured nothing is not a low score.* Warnings from `scaffold.py`
are shown against the rows they belong to and counted at the top, because a
model that never emitted a tool call will otherwise sit at the bottom of a
leaderboard looking merely bad.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import stats
from .ui import STYLESHEET

DEFAULT_ROOT = Path("runs")


# --------------------------------------------------------------------------- #
# Reading the runs
# --------------------------------------------------------------------------- #


@dataclass
class RunRecord:
    """One saved run directory, reduced to what an aggregate view needs."""

    path: str
    agent: str
    benchmark: str
    saved_at: str
    mean: float
    cost: float
    results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.get("warnings"))

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.get("status") != "ok")

    @property
    def unsafe(self) -> int:
        return sum(1 for r in self.results if not r["score"]["safe"])


def _benchmark_of(meta: dict[str, Any]) -> tuple[str, int, int]:
    """Name, instances run, instances available.

    Runs saved before the benchmark layer existed have no such key; they came
    from the local task directory, and saying so is better than an empty cell.
    """
    block = meta.get("benchmark") or {}
    name = block.get("name") or "local"
    return name, int(block.get("ran") or meta.get("runs") or 0), int(
        block.get("instances") or 0
    )


def collect(root: Path | str = DEFAULT_ROOT) -> list[RunRecord]:
    """Every run under `root`, newest first.

    A directory that cannot be read is skipped rather than fatal — half the
    value of this page is looking at a collection that includes an interrupted
    run, and refusing to render because one file is malformed would be exactly
    backwards.
    """
    root = Path(root)
    records: list[RunRecord] = []
    for path in sorted(root.glob("*/results.json")):
        try:
            payload = json.loads(path.read_text())
            meta = payload["meta"]
        except Exception:  # noqa: BLE001
            continue
        name, _, _ = _benchmark_of(meta)
        records.append(
            RunRecord(
                path=str(path.parent.name),
                agent=str(meta.get("agent", "unknown")),
                benchmark=name,
                saved_at=str(meta.get("saved_at", "")),
                mean=float(meta.get("mean_overall") or 0.0),
                cost=float(meta.get("total_cost_usd") or 0.0),
                results=payload.get("results", []),
            )
        )
    records.sort(key=lambda r: r.saved_at, reverse=True)
    return records


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _scores(results: list[dict[str, Any]]) -> list[float]:
    return [r["score"]["overall"] for r in results]


def leaderboard(records: list[RunRecord]) -> list[dict[str, Any]]:
    """One row per agent and benchmark, over every run of that pair."""
    groups: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        groups.setdefault((record.agent, record.benchmark), []).append(record)

    rows = []
    for (agent, benchmark), runs in groups.items():
        results = [r for run in runs for r in run.results]
        scores = _scores(results)
        tasks = {r["task_id"] for r in results}
        rows.append({
            "agent": agent,
            "benchmark": benchmark,
            "runs": len(runs),
            "n": len(results),
            "tasks": len(tasks),
            "mean": statistics.fmean(scores) if scores else 0.0,
            "stdev": statistics.stdev(scores) if len(scores) > 1 else None,
            # The band the evidence supports, not just the point. Two rows a
            # few points apart with overlapping intervals are not ranked, and
            # a table that shows only the means says otherwise.
            "low": stats.interval(scores).low if scores else 0.0,
            "high": stats.interval(scores).high if scores else 0.0,
            "cost": sum(run.cost for run in runs),
            "warnings": sum(run.warnings for run in runs),
            "errors": sum(run.errors for run in runs),
            "unsafe": sum(run.unsafe for run in runs),
            "latest": max(run.saved_at for run in runs),
            # The all-time mean answers "how has this pair done"; the latest
            # answers "where does it stand now". They diverge whenever a task
            # or a verifier changed underneath, which happens — one of the
            # runs in this very directory scored 0.50 on a task that a bug
            # was zeroing, and averaging it with the fixed run describes
            # neither.
            "latest_mean": statistics.fmean(
                _scores(max(runs, key=lambda r: r.saved_at).results)
            ) if any(run.results for run in runs) else None,
            "where": [run.path for run in runs],
        })
    rows.sort(key=lambda r: (-r["mean"], r["agent"]))
    return rows


#: Replaying a task's own reference solution. It scores 1.00 by construction —
#: that is the point of it, proof the task is solvable and the verifier
#: satisfiable — and exactly why it must not count as a model when measuring
#: whether a task separates models. Counting it makes every task look
#: discriminating, which is the opposite of the truth.
REFERENCE_AGENT = "gold"


def is_model(agent: str) -> bool:
    return agent != REFERENCE_AGENT


def by_task(records: list[RunRecord]) -> list[dict[str, Any]]:
    """Per task, what each agent scored, and whether the task tells you
    anything.

    Two questions, because they need different amounts of evidence.

    *Spread* — best model's mean minus worst — answers "does this separate the
    models I have run". It needs at least two models.

    *Headroom* — one minus the best model's mean — answers "can this still show
    an improvement". It needs only one, and it is what catches a saturated
    suite before you have a second frontier model to compare against. A task
    the best model already solves perfectly cannot rank anything above it.
    """
    per_task: dict[str, dict[str, list[float]]] = {}
    benchmarks: dict[str, str] = {}
    for record in records:
        for result in record.results:
            task = result["task_id"]
            benchmarks[task] = record.benchmark
            per_task.setdefault(task, {}).setdefault(record.agent, []).append(
                result["score"]["overall"]
            )

    rows = []
    for task, agents in per_task.items():
        means = {a: statistics.fmean(v) for a, v in agents.items()}
        models = {a: m for a, m in means.items() if is_model(a)}
        values = list(models.values())
        rows.append({
            "task": task,
            "benchmark": benchmarks.get(task, ""),
            "agents": means,
            "n": sum(len(v) for v in agents.values()),
            "spread": (max(values) - min(values)) if len(values) > 1 else None,
            "headroom": (1.0 - max(values)) if values else None,
            "solvable": means.get(REFERENCE_AGENT),
            "mean": statistics.fmean(values) if values else 0.0,
        })
    # Least informative first: the tasks that separate nothing are the finding,
    # so they should not be buried at the bottom of a long table.
    rows.sort(key=lambda r: (r["spread"] if r["spread"] is not None else 1e9,
                             r["headroom"] if r["headroom"] is not None else 1e9,
                             -r["mean"]))
    return rows


def head_to_head(records: list[RunRecord]) -> list[dict[str, Any]]:
    """Every pair of models that share a benchmark, compared on shared tasks.

    Paired rather than by marginal means: instance difficulty dominates the
    variance, and both models faced the same instances, so differencing per
    instance removes it exactly.
    """
    scores: dict[tuple[str, str], dict[str, list[float]]] = {}
    for record in records:
        if not is_model(record.agent):
            continue
        bucket = scores.setdefault((record.benchmark, record.agent), {})
        for result in record.results:
            bucket.setdefault(result["task_id"], []).append(
                result["score"]["overall"]
            )

    rows = []
    benchmarks = sorted({b for b, _ in scores})
    for benchmark in benchmarks:
        agents = sorted(a for b, a in scores if b == benchmark)
        for i, left in enumerate(agents):
            for right in agents[i + 1:]:
                outcome = stats.compare(
                    scores[(benchmark, left)], scores[(benchmark, right)],
                    left, right,
                )
                if not outcome.paired:
                    continue
                rows.append({
                    "benchmark": benchmark,
                    "left": left, "right": right,
                    "paired": outcome.paired,
                    "delta": outcome.difference.point,
                    "low": outcome.difference.low,
                    "high": outcome.difference.high,
                    "wins": outcome.wins, "losses": outcome.losses,
                    "ties": outcome.ties,
                    "decisive": outcome.decisive,
                    "verdict": outcome.verdict(),
                    "needed": stats.sample_size_for(
                        max(0.05, abs(outcome.difference.point))
                    ),
                })
    rows.sort(key=lambda r: (-abs(r["delta"]), r["benchmark"]))
    return rows


def build(root: Path | str = DEFAULT_ROOT) -> dict[str, Any]:
    records = collect(root)
    agents = sorted({r.agent for r in records})
    return {
        # Deliberately no absolute path to the runs directory. Nothing on the
        # page renders it, and a self-contained file that can be handed to
        # someone should not carry the filesystem of the machine that made it.
        "runs": [
            {
                "path": r.path, "agent": r.agent, "benchmark": r.benchmark,
                "saved_at": r.saved_at, "mean": r.mean, "cost": r.cost,
                "n": r.n, "warnings": r.warnings, "errors": r.errors,
                "unsafe": r.unsafe,
            }
            for r in records
        ],
        "leaderboard": leaderboard(records),
        "head_to_head": head_to_head(records),
        "tasks": by_task(records),
        "agents": [a for a in agents if is_model(a)],
        "reference_agent": REFERENCE_AGENT,
        "totals": {
            "runs": len(records),
            "measurements": sum(r.n for r in records),
            "cost": sum(r.cost for r in records),
            "agents": len(agents),
            "benchmarks": len({r.benchmark for r in records}),
            "warnings": sum(r.warnings for r in records),
        },
    }


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #

EXTRA_CSS = """
.wrap { padding:1.5rem 1.75rem 5rem; max-width:1400px; margin:0 auto; }
.totals { display:flex; gap:2.25rem; flex-wrap:wrap; padding-bottom:1.25rem;
  border-bottom:1px solid var(--rule); margin-bottom:.5rem; }
.totals .n { font-family:var(--mono); font-size:1.45rem; letter-spacing:-.02em; }
.totals .n.warn { color:var(--partial); }

.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th { text-align:left; font-family:var(--display); font-size:10px;
  letter-spacing:.16em; text-transform:uppercase; color:var(--muted);
  font-weight:600; padding:.5rem .6rem; border-bottom:1px solid var(--rule);
  white-space:nowrap; }
td { padding:.5rem .6rem; border-bottom:1px solid var(--rule-soft);
  vertical-align:baseline; }
td.num { font-family:var(--mono); font-variant-numeric:tabular-nums;
  text-align:right; white-space:nowrap; }
tr:hover td { background:var(--rule-soft); }
.agent { font-family:var(--mono); font-size:12.5px; }
.tag { font-size:10.5px; font-family:var(--mono); color:var(--muted);
  border:1px solid var(--rule); border-radius:99px; padding:.05rem .45rem; }
.pass { color:var(--pass); } .mid { color:var(--partial); } .low { color:var(--fail); }
.subtle { color:var(--muted); }

/* A bar behind the number, so a column of means is scannable as a shape. */
.bar { position:relative; min-width:82px; }
.bar i { position:absolute; left:.6rem; right:.6rem; bottom:2px; height:2px;
  background:var(--rule-soft); border-radius:2px; }
.bar i b { display:block; height:100%; border-radius:2px; background:var(--pass); }
.bar.mid i b { background:var(--partial); } .bar.low i b { background:var(--fail); }

.flag-warn { color:var(--partial); font-size:11px; font-family:var(--mono); }
.saturated td { background:color-mix(in srgb, var(--partial) 7%, transparent); }
.note-row { font-size:12.5px; color:var(--muted); margin:.5rem 0 1.25rem; }
a.run { color:var(--accent); text-decoration:none; font-family:var(--mono);
  font-size:12px; }
a.run:hover { text-decoration:underline; }
.empty { color:var(--muted); font-style:italic; }
.legend { display:flex; gap:1.25rem; flex-wrap:wrap; font-size:11.5px;
  color:var(--muted); margin-top:.6rem; }
.controls { display:flex; gap:1.25rem; align-items:center; flex-wrap:wrap;
  margin-bottom:.75rem; font-size:12px; color:var(--muted); }
.controls label { display:flex; gap:.4rem; align-items:center; cursor:pointer; }
.controls select { font:inherit; font-family:var(--mono); font-size:12px;
  padding:.2rem .4rem; border:1px solid var(--rule); border-radius:5px;
  background:var(--card); color:var(--ink); }
td.untested { color:var(--muted); font-style:italic; }
"""

_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agenteval — all runs</title>
<style>
__STYLESHEET__
__EXTRA__
</style>
</head><body>
<header>
  <h1>agenteval</h1>
  <span class="agent">all runs</span>
  <span class="facts" id="facts"></span>
  <button id="theme">Theme</button>
</header>
<div class="wrap">
  <div class="totals" id="totals"></div>
  <section class="section">
    <span class="eyebrow">Which tasks separate models</span>
    <p class="note-row" id="discrimination-note"></p>
    <div class="controls">
      <label>benchmark <select id="filter-benchmark"></select></label>
      <label><input type="checkbox" id="filter-dull"> hide tasks that still inform</label>
      <span class="subtle" id="showing"></span>
    </div>
    <div class="scroll"><table id="tasks"></table></div>
    <div class="legend">
      <span><b>spread</b> = best model's mean minus worst — does this task separate the models you have run?</span>
      <span><b>headroom</b> = 1 − best model's mean — can it still show an improvement?</span>
      <span><b>gold</b> = the reference solution, proof the task is solvable; not a model, so it counts toward neither</span>
    </div>
  </section>
  <section class="section">
    <span class="eyebrow">Leaderboard</span>
    <p class="note-row">One row per agent and benchmark, over every run of that
      pair. A subset and a full run are different claims — the task count says
      which.</p>
    <div class="scroll"><table id="board"></table></div>
  </section>
  <section class="section" id="h2h-section">
    <span class="eyebrow">Head to head</span>
    <p class="note-row">Paired over the instances both models attempted.
      Comparing marginal means throws away the pairing, and instance difficulty
      is the largest source of variance here.</p>
    <div class="scroll"><table id="h2h"></table></div>
  </section>
  <section class="section">
    <span class="eyebrow">Every run</span>
    <div class="scroll"><table id="runs"></table></div>
  </section>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const esc = s => String(s == null ? '' : s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
  .replace(/"/g,'&quot;');
const band = v => v >= 0.85 ? 'pass' : v >= 0.5 ? 'mid' : 'low';
const fmt = v => v === null || v === undefined ? '—' : v.toFixed(2);
const when = s => s ? esc(String(s).slice(0, 16).replace('T', ' ')) : '—';

// A number with a bar behind it reads as a shape down a column, which is how
// you spot a model that is ahead everywhere versus ahead in one place.
const scoreCell = v => v === null || v === undefined
  ? '<td class="num subtle">—</td>'
  : `<td class="num bar ${band(v)}">${fmt(v)}<i><b style="width:${
      Math.max(0, Math.min(1, v)) * 100}%"></b></i></td>`;

const t = DATA.totals;
document.getElementById('facts').innerHTML =
  `<b>${t.runs}</b> runs · <b>${t.measurements}</b> measurements · ` +
  `<b>${t.agents}</b> agents · <b>${t.benchmarks}</b> benchmarks`;

document.getElementById('totals').innerHTML = [
  ['Runs', t.runs, ''],
  ['Measurements', t.measurements, ''],
  ['Agents', t.agents, ''],
  ['Spend', '$' + t.cost.toFixed(2), ''],
  ['Suspect runs', t.warnings, t.warnings ? 'warn' : ''],
].map(([label, value, cls]) =>
  `<div class="score"><span class="eyebrow">${label}</span>
   <div class="n ${cls}">${esc(value)}</div></div>`).join('');

/* ---- which tasks separate models ------------------------------------- */
const agents = DATA.agents;
const rows = DATA.tasks;
const compared = rows.filter(r => r.spread !== null);
const flat = compared.filter(r => r.spread < 0.05);
const measured = rows.filter(r => r.headroom !== null);
const saturated = measured.filter(r => r.headroom < 0.05);
const parts = [];
if (compared.length) {
  parts.push(`${flat.length} of ${compared.length} tasks run by more than one `
    + `model separate them by less than 0.05.`);
} else if (measured.length) {
  parts.push('Only one model has run, so spread cannot be computed yet — '
    + 'headroom is the column to read.');
}
if (measured.length) {
  parts.push(`${saturated.length} of ${measured.length} are already solved by `
    + `the best model, so they cannot rank anything above it.`);
}
document.getElementById('discrimination-note').textContent =
  parts.join(' ') || 'No runs to compare yet.';

// A row is shaded when it has stopped informing: either the models it has
// seen are indistinguishable on it, or the best of them already solves it.
const dull = r => (r.spread !== null && r.spread < 0.05)
               || (r.headroom !== null && r.headroom < 0.05);

// SWE-bench Lite alone is three hundred rows, so the table needs narrowing
// before it needs prettifying.
const benchmarks = [...new Set(rows.map(r => r.benchmark))].sort();
const select = document.getElementById('filter-benchmark');
select.innerHTML = ['<option value="">all</option>']
  .concat(benchmarks.map(b => `<option value="${esc(b)}">${esc(b)}</option>`)).join('');

function drawTasks() {
  const wanted = select.value;
  const onlyDull = document.getElementById('filter-dull').checked;
  const shown = rows.filter(r =>
    (!wanted || r.benchmark === wanted) && (!onlyDull || dull(r)));
  document.getElementById('showing').textContent =
    `${shown.length} of ${rows.length} tasks`;
  document.getElementById('tasks').innerHTML =
    `<thead><tr><th>task</th><th>benchmark</th>` +
    agents.map(a => `<th style="text-align:right">${esc(a)}</th>`).join('') +
    `<th style="text-align:right">gold</th>
     <th style="text-align:right">spread</th>
     <th style="text-align:right">headroom</th>
     <th style="text-align:right">n</th></tr></thead><tbody>` +
    shown.map(r => {
      // Never attempted by a model: gold proves it is solvable, and nothing
      // else is known. Saying so beats a row of dashes that reads like a dud.
      const untested = r.headroom === null;
      return `<tr class="${dull(r) ? 'saturated' : ''}">
        <td class="${untested ? 'untested' : ''}">${esc(r.task)}</td>
        <td><span class="tag">${esc(r.benchmark)}</span></td>` +
        agents.map(a => scoreCell(r.agents[a])).join('') +
        `<td class="num subtle">${r.solvable === undefined || r.solvable === null
           ? '—' : r.solvable.toFixed(2)}</td>` +
        (untested
          ? `<td class="num untested" colspan="2">no model has run it</td>`
          : `<td class="num ${r.spread === null ? 'subtle' : r.spread < 0.05 ? 'mid' : ''}">${
               r.spread === null ? '—' : r.spread.toFixed(2)}</td>
             <td class="num ${r.headroom < 0.05 ? 'mid' : ''}">${
               r.headroom.toFixed(2)}</td>`) +
        `<td class="num subtle">${r.n}</td></tr>`;
    }).join('') + `</tbody>`;
}
select.onchange = drawTasks;
document.getElementById('filter-dull').onchange = drawTasks;
drawTasks();

/* ---- leaderboard ------------------------------------------------------ */
document.getElementById('board').innerHTML =
  `<thead><tr><th>agent</th><th>benchmark</th>
   <th style="text-align:right">all-time</th>
   <th style="text-align:right">95%</th>
   <th style="text-align:right">latest</th>
   <th style="text-align:right">tasks</th><th style="text-align:right">n</th>
   <th style="text-align:right">runs</th><th style="text-align:right">cost</th>
   <th>trust</th><th>latest</th></tr></thead><tbody>` +
  DATA.leaderboard.map(r => `<tr>
    <td class="agent">${esc(r.agent)}</td>
    <td><span class="tag">${esc(r.benchmark)}</span></td>` +
    scoreCell(r.mean) +
    `<td class="num subtle">[${r.low.toFixed(2)}, ${r.high.toFixed(2)}]</td>` +
    scoreCell(r.latest_mean) +
    `<td class="num">${r.tasks}</td>
     <td class="num">${r.n}</td>
     <td class="num subtle">${r.runs}</td>
     <td class="num">$${r.cost.toFixed(2)}</td>
     <td>${[
       r.warnings ? `<span class="flag-warn">${r.warnings} suspect</span>` : '',
       r.unsafe ? `<span class="flag unsafe">${r.unsafe} unsafe</span>` : '',
       r.errors ? `<span class="flag-warn">${r.errors} errored</span>` : '',
     ].filter(Boolean).join(' ') || '<span class="subtle">—</span>'}</td>
     <td class="subtle">${when(r.latest)}</td></tr>`).join('') +
  `</tbody>`;

/* ---- head to head ----------------------------------------------------- */
const h2h = DATA.head_to_head || [];
if (!h2h.length) {
  document.getElementById('h2h-section').style.display = 'none';
} else {
  document.getElementById('h2h').innerHTML =
    `<thead><tr><th>benchmark</th><th>models</th>
     <th style="text-align:right">difference</th>
     <th style="text-align:right">95%</th>
     <th style="text-align:right">shared</th>
     <th style="text-align:right">w / l / t</th>
     <th>verdict</th></tr></thead><tbody>` +
    h2h.map(r => `<tr>
      <td><span class="tag">${esc(r.benchmark)}</span></td>
      <td class="agent">${esc(r.left)} <span class="subtle">vs</span> ${esc(r.right)}</td>
      <td class="num">${r.delta >= 0 ? '+' : ''}${r.delta.toFixed(2)}</td>
      <td class="num subtle">[${r.low.toFixed(2)}, ${r.high.toFixed(2)}]</td>
      <td class="num">${r.paired}</td>
      <td class="num subtle">${r.wins} / ${r.losses} / ${r.ties}</td>
      <td class="${r.decisive ? 'pass' : 'subtle'}">${esc(r.verdict)}${
        r.decisive ? '' :
        ` <span class="subtle">(≈${r.needed} shared instances would settle it)</span>`
      }</td></tr>`).join('') + `</tbody>`;
}

/* ---- every run -------------------------------------------------------- */
document.getElementById('runs').innerHTML =
  `<thead><tr><th>when</th><th>agent</th><th>benchmark</th>
   <th style="text-align:right">mean</th><th style="text-align:right">n</th>
   <th style="text-align:right">cost</th><th>trust</th><th>report</th>
   </tr></thead><tbody>` +
  DATA.runs.map(r => `<tr>
    <td class="subtle">${when(r.saved_at)}</td>
    <td class="agent">${esc(r.agent)}</td>
    <td><span class="tag">${esc(r.benchmark)}</span></td>` +
    scoreCell(r.mean) +
    `<td class="num">${r.n}</td>
     <td class="num">$${r.cost.toFixed(2)}</td>
     <td>${[
       r.warnings ? `<span class="flag-warn">${r.warnings} suspect</span>` : '',
       r.unsafe ? `<span class="flag unsafe">${r.unsafe} unsafe</span>` : '',
       r.errors ? `<span class="flag-warn">${r.errors} errored</span>` : '',
     ].filter(Boolean).join(' ') || '<span class="subtle">—</span>'}</td>
     <td><a class="run" href="${esc(r.path)}/report.html">${esc(r.path)}</a></td>
     </tr>`).join('') +
  `</tbody>`;

if (!DATA.runs.length) {
  document.querySelector('.wrap').innerHTML =
    '<p class="empty">No runs found. Try <code>agenteval run --gold</code>.</p>';
}

/* ---- theme ------------------------------------------------------------ */
const root = document.documentElement;
document.getElementById('theme').onclick = () => {
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  const now = root.getAttribute('data-theme') || (dark ? 'dark' : 'light');
  root.setAttribute('data-theme', now === 'dark' ? 'light' : 'dark');
};
</script>
</body></html>
"""


def render(payload: dict[str, Any]) -> str:
    # Same escaping as the run report: the payload carries task ids and agent
    # names, and a literal `</script>` in any of them would close the block
    # early and execute whatever followed.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return (
        _TEMPLATE.replace("__STYLESHEET__", STYLESHEET)
        .replace("__EXTRA__", EXTRA_CSS)
        .replace("__PAYLOAD__", data)
    )


def write(root: Path | str = DEFAULT_ROOT, out: Path | str | None = None) -> Path:
    """Build the dashboard beside the runs it describes.

    Written into the runs directory by default so the per-run links resolve as
    plain relative paths and the whole thing works from `file://`.
    """
    payload = build(root)
    path = Path(out) if out else Path(root) / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(payload), encoding="utf-8")
    return path
