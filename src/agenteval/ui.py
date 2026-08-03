"""The single-file run explorer.

Kept apart from `report.py` — that module owns aggregation, persistence and the
terminal view; this one owns the HTML. They share the saved payload and nothing
else.

Design brief, since the reasoning is not obvious from the markup:

The page has one job — find the moment an agent went wrong and show how it was
graded. The CLI already prints scores, so a scoreboard would be redundant. What
it renders instead is a *recording*: a ruled spine of numbered steps where
ordinary calls are hairlines and blocked or errored calls punch through the
rule, so deviations are found by scanning rather than reading.

The artifacts panel is the other half. Emails, documents and tickets the agent
produced currently exist only as JSON inside tool arguments, which is unreadable
precisely where reading matters most — it is the material the rubric judged.
They are reconstructed from the call log and rendered as what they are.

Everything is inlined and the payload is embedded, so the file works from
file://, offline, forever, with no build step and no server.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any
#: Foundations — colour, type and elevation. The design-system project and the
#: report are generated from this one block, so a palette change in one cannot
#: silently fail to reach the other.
TOKENS = """\
:root {
  --paper:#F7F8FA; --card:#FFFFFF; --ink:#14202E; --muted:#5C6B7F;
  --rule:#D3DAE3; --rule-soft:#E7ECF2; --accent:#2F5FD0;
  --pass:#1F7A5C; --partial:#B26B00; --fail:#B3261E;
  --shadow:0 1px 2px rgba(20,32,46,.05), 0 8px 24px -12px rgba(20,32,46,.18);
  --display:"Futura","Avenir Next","Century Gothic",system-ui,sans-serif;
  --body:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
  --mono:"SF Mono",Menlo,"JetBrains Mono",ui-monospace,monospace;
}
/* Dark values live once, applied either by preference or by the toggle.
   The toggle must win in both directions, so it is a separate rule. */
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:#0E1420; --card:#151D2B; --ink:#E6EBF2; --muted:#8E9DB3;
    --rule:#26313F; --rule-soft:#1D2634; --accent:#7BA0F5;
    --pass:#4FBF92; --partial:#E0A54A; --fail:#F2857C;
    --shadow:0 1px 2px rgba(0,0,0,.35), 0 8px 24px -12px rgba(0,0,0,.6);
    color-scheme:dark;
  }
}
:root[data-theme="dark"] {
  --paper:#0E1420; --card:#151D2B; --ink:#E6EBF2; --muted:#8E9DB3;
  --rule:#26313F; --rule-soft:#1D2634; --accent:#7BA0F5;
  --pass:#4FBF92; --partial:#E0A54A; --fail:#F2857C;
  --shadow:0 1px 2px rgba(0,0,0,.35), 0 8px 24px -12px rgba(0,0,0,.6);
  color-scheme:dark;
}
"""

#: Reset and the two typographic utilities every component leans on.
BASE = """\
* { box-sizing:border-box; }
body {
  margin:0; background:var(--paper); color:var(--ink);
  font:15px/1.55 var(--body); -webkit-font-smoothing:antialiased;
}
.eyebrow {
  font-family:var(--display); font-size:10px; letter-spacing:.18em;
  text-transform:uppercase; color:var(--muted); font-weight:600;
}
.mono { font-family:var(--mono); font-variant-numeric:tabular-nums; }
"""

#: Page chrome. Specific to the explorer's two-pane shell, so the design-system
#: specimens deliberately leave it out.
CHROME = """\
/* ---- header ---------------------------------------------------------- */
header {
  border-bottom:1px solid var(--rule); background:var(--card);
  padding:1.1rem 1.5rem; display:flex; gap:1.5rem; align-items:baseline;
  flex-wrap:wrap; position:sticky; top:0; z-index:5;
}
header h1 {
  font-family:var(--display); font-size:13px; letter-spacing:.22em;
  text-transform:uppercase; margin:0; font-weight:700;
}
header .agent { font-family:var(--mono); font-size:13px; color:var(--accent); }
header .facts { color:var(--muted); font-size:13px; margin-left:auto; }
header .facts b { color:var(--ink); font-weight:600; }
#theme {
  border:1px solid var(--rule); background:transparent; color:var(--muted);
  font:inherit; font-size:11px; padding:.25rem .6rem; border-radius:99px;
  cursor:pointer;
}
#theme:hover { color:var(--ink); border-color:var(--muted); }

/* ---- layout ---------------------------------------------------------- */
.app { display:grid; grid-template-columns:270px minmax(0,1fr); min-height:80vh; }
@media (max-width:820px) { .app { grid-template-columns:1fr; } }

nav {
  border-right:1px solid var(--rule); padding:1rem .75rem 3rem;
}
@media (max-width:820px) { nav { border-right:none; border-bottom:1px solid var(--rule); } }
nav .head { display:flex; align-items:center; gap:.5rem; padding:0 .5rem .6rem; }
nav label { font-size:11px; color:var(--muted); display:flex; gap:.35rem;
  align-items:center; cursor:pointer; margin-left:auto; }


main { padding:1.5rem 1.75rem 5rem; min-width:0; }
@media (max-width:820px) { main { padding:1.25rem 1rem 4rem; } }
.section { margin-top:2.25rem; }
.section > .eyebrow { display:block; margin-bottom:.7rem; }
"""

#: The component library: score display, trace spine, verdict rows, artifact
#: cards. Shared verbatim with the design-system specimens.
COMPONENTS = """\
/* ---- run index entry -------------------------------------------------- */
.entry {
  display:block; width:100%; text-align:left; background:none; cursor:pointer;
  border:none; border-radius:7px; padding:.5rem .6rem; margin-bottom:1px;
  color:inherit; font:inherit;
}
.entry:hover { background:var(--rule-soft); }
.entry[aria-current="true"] { background:var(--accent); color:#fff; }
.entry[aria-current="true"] .sub,
.entry[aria-current="true"] .val { color:rgba(255,255,255,.85); }
.entry .top { display:flex; align-items:baseline; gap:.5rem; }
.entry .name { font-size:13px; font-weight:500; }
.entry .val { margin-left:auto; font-family:var(--mono); font-size:12px;
  color:var(--muted); }
.entry .sub { font-size:10.5px; color:var(--muted); font-family:var(--mono); }

/* score meter */
.meter { height:3px; border-radius:2px; background:var(--rule-soft); margin-top:.35rem; }
.entry[aria-current="true"] .meter { background:rgba(255,255,255,.25); }
.meter i { display:block; height:100%; border-radius:2px; background:var(--pass); }
.meter.mid i { background:var(--partial); } .meter.low i { background:var(--fail); }
.entry[aria-current="true"] .meter i { background:#fff; }

/* ---- run header ------------------------------------------------------ */
.runhead h2 { font-size:1.35rem; margin:.15rem 0 .5rem; font-weight:600;
  letter-spacing:-.01em; }
.scores { display:flex; gap:1.75rem; flex-wrap:wrap; align-items:flex-end;
  padding-bottom:1rem; border-bottom:1px solid var(--rule); }
.score .eyebrow { display:block; margin-bottom:.15rem; }
.score .n { font-family:var(--mono); font-size:1.45rem; letter-spacing:-.02em; }
.score .n.pass { color:var(--pass); } .score .n.mid { color:var(--partial); }
.score .n.low { color:var(--fail); }
.flag { font-family:var(--display); font-size:11px; letter-spacing:.14em;
  padding:.25rem .5rem; border-radius:4px; text-transform:uppercase; font-weight:700; }
.flag.safe { color:var(--pass); background:color-mix(in srgb, var(--pass) 12%, transparent); }
.flag.unsafe { color:var(--fail); background:color-mix(in srgb, var(--fail) 14%, transparent); }
.note { color:var(--partial); font-size:13px; margin-top:.6rem; font-family:var(--mono); }

.repeats { display:flex; gap:.3rem; margin-top:.9rem; }
.repeats button {
  font:inherit; font-family:var(--mono); font-size:11px; padding:.2rem .55rem;
  border:1px solid var(--rule); background:transparent; color:var(--muted);
  border-radius:5px; cursor:pointer;
}
.repeats button[aria-pressed="true"] { border-color:var(--accent);
  color:var(--accent); }

/* ---- the trace ------------------------------------------------------- */
/* A ruled spine with a node per step. Ordinary calls are faint dots on the
   rule; blocked and errored calls swell into filled squares that break it, so
   the run is scanned for deviations rather than read top to bottom.
   Gutter geometry is in px on purpose — the node has to land dead centre on a
   1px rule, and rem arithmetic makes that a guess. */
.trace { list-style:none; margin:0; padding:0 0 0 56px; position:relative; }
.trace::before {
  content:""; position:absolute; left:40px; top:.5rem; bottom:.5rem;
  width:1px; background:var(--rule);
}
.step {
  position:relative; padding:.3rem .4rem .35rem .5rem; cursor:pointer;
  border-radius:0 6px 6px 0;
}
.step:hover, .step.open { background:var(--rule-soft); }
.step .num {
  position:absolute; left:-56px; top:.36rem; width:30px; text-align:right;
  font-family:var(--mono); font-size:11px; color:var(--muted);
}
/* the node, centred on the 40–41px rule */
.step::before {
  content:""; position:absolute; left:-18px; top:.62rem; width:5px; height:5px;
  border-radius:50%; background:var(--rule); box-shadow:0 0 0 3px var(--paper);
}
.step .tool { font-family:var(--mono); font-size:12.5px; font-weight:500; }
.step .args { font-family:var(--mono); font-size:11.5px; color:var(--muted);
  margin-left:.5rem; }
/* One line by default: when scanning for the wrong call, the tool name and its
   arguments are the signal and the payload is noise. Click to open it. */
.step .out {
  font-family:var(--mono); font-size:11px; color:var(--muted); margin-top:.1rem;
  white-space:pre-wrap; word-break:break-word; overflow:hidden;
  display:-webkit-box; -webkit-line-clamp:1; -webkit-box-orient:vertical;
  opacity:.85;
}
.step.open .out { -webkit-line-clamp:unset; display:block; opacity:1; }

/* punch-outs */
.step[data-flag="blocked"], .step[data-flag="error"] {
  box-shadow:inset 2px 0 0 var(--fail);
}
.step[data-flag="error"] { box-shadow:inset 2px 0 0 var(--partial); }
.step[data-flag="blocked"] .num, .step[data-flag="error"] .num {
  color:var(--ink); font-weight:600;
}
.step[data-flag="blocked"]::before, .step[data-flag="error"]::before {
  left:-20px; top:.52rem; width:9px; height:9px; border-radius:2px;
  background:var(--fail);
}
.step[data-flag="error"]::before { background:var(--partial); }
.step[data-flag="blocked"] .tool { color:var(--fail); }
.step[data-flag="error"] .tool { color:var(--partial); }
.tag { font-family:var(--display); font-size:9.5px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--fail); margin-left:.5rem; font-weight:700; }
.step[data-flag="error"] .tag { color:var(--partial); }

/* ---- verdict --------------------------------------------------------- */
.checks { list-style:none; margin:0; padding:0; }
.checks li { display:flex; gap:.6rem; padding:.3rem 0; align-items:baseline;
  border-bottom:1px solid var(--rule-soft); }
.checks li:last-child { border-bottom:none; }
.checks .mark { font-family:var(--mono); font-size:12px; width:1rem; flex:none; }
.checks .pass .mark { color:var(--pass); } .checks .fail .mark { color:var(--fail); }
.checks .why { color:var(--muted); font-size:13px; }
.checks .w { margin-left:auto; font-family:var(--mono); font-size:10.5px;
  color:var(--muted); flex:none; }
.rubric li .mark { color:var(--partial); }
.rubric li.full .mark { color:var(--pass); }
.violations { list-style:none; margin:0; padding:0; }
.violations li {
  border-left:2px solid var(--fail); padding:.4rem .7rem; margin-bottom:.4rem;
  background:color-mix(in srgb, var(--fail) 7%, transparent);
  font-size:13.5px; border-radius:0 5px 5px 0;
}

/* ---- artifacts ------------------------------------------------------- */
/* What the agent actually produced, rendered as what it is rather than as
   JSON inside a tool argument — this is the material the rubric graded. */
.artifact {
  border:1px solid var(--rule); border-radius:9px; background:var(--card);
  padding:.9rem 1rem; margin-bottom:.8rem; box-shadow:var(--shadow);
}
.artifact .kind { display:flex; gap:.5rem; align-items:baseline; margin-bottom:.5rem; }
.artifact .kind .eyebrow { color:var(--accent); }
.artifact .to { font-family:var(--mono); font-size:11.5px; color:var(--muted); }
.artifact h4 { margin:0 0 .4rem; font-size:14.5px; font-weight:600; }
.artifact .doc {
  white-space:pre-wrap; font-size:13.5px; line-height:1.6; margin:0;
  font-family:var(--body); color:var(--ink); max-height:22rem; overflow:auto;
}
.empty { color:var(--muted); font-size:13.5px; font-style:italic; }

/* ---- task definition -------------------------------------------------- */
/* Material the agent was *given* is flat and left-ruled; material it produced
   is a raised card. Elevation means authorship, which is the distinction that
   decides whether a failure belongs to the agent or to the task. */
.switch { display:flex; gap:.25rem; margin:1rem 0 0; }
.switch button {
  font:inherit; font-family:var(--display); font-size:10px; letter-spacing:.14em;
  text-transform:uppercase; font-weight:600; padding:.35rem .75rem;
  border:1px solid var(--rule); background:transparent; color:var(--muted);
  border-radius:5px; cursor:pointer;
}
.switch button[aria-pressed="true"] {
  background:var(--ink); color:var(--paper); border-color:var(--ink);
}
.brief {
  border-left:2px solid var(--accent); padding:.15rem 0 .15rem 1rem; margin:0;
  white-space:pre-wrap; font-size:14.5px; line-height:1.6;
}
.given { border-left:2px solid var(--rule); padding:.2rem 0 .2rem 1rem;
  margin-bottom:1.1rem; }
.given .kind { display:flex; gap:.5rem; align-items:baseline; margin-bottom:.3rem; }
.given .to { font-family:var(--mono); font-size:11.5px; color:var(--muted); }
.given h4 { margin:0 0 .3rem; font-size:14px; font-weight:600; }
.given .doc {
  white-space:pre-wrap; font-size:13px; line-height:1.6; margin:0;
  color:var(--muted); max-height:15rem; overflow:auto;
}
.chips { display:flex; flex-wrap:wrap; gap:.35rem; }
.chip {
  font-family:var(--mono); font-size:11px; padding:.2rem .5rem;
  border:1px solid var(--rule); border-radius:4px; color:var(--muted);
}
.chip.deny {
  color:var(--fail);
  border-color:color-mix(in srgb, var(--fail) 35%, transparent);
}
.tablewrap { overflow-x:auto; }
.records { border-collapse:collapse; font-family:var(--mono); font-size:11.5px; }
.records th {
  text-align:left; font-family:var(--display); font-size:9px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); font-weight:600;
  padding:.3rem 1rem .3rem 0; border-bottom:1px solid var(--rule);
  white-space:nowrap;
}
.records td {
  padding:.3rem 1rem .3rem 0; border-bottom:1px solid var(--rule-soft);
  vertical-align:top; max-width:26rem;
}
/* Seeded records carry the material a task turns on — an injected note in an
   expense, the body of a ticket. Clamped so the table stays a table, but the
   whole value is present and a click opens the row: truncating it away would
   hide the very thing a reviewer opened this view to read. */
.records tr[data-row] { cursor:pointer; }
.records tr[data-row]:hover td { background:var(--rule-soft); }
.records td > span {
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
  overflow:hidden; white-space:pre-wrap;
}
.records tr.open td > span { -webkit-line-clamp:unset; display:block; }
.source pre {
  font-family:var(--mono); font-size:11.5px; line-height:1.55; margin:0;
  background:var(--card); border:1px solid var(--rule); border-radius:8px;
  padding:.9rem 1rem; overflow:auto; max-height:32rem;
}
"""

#: Focus ring and motion preference — part of the quality floor, not the look.
UTIL = """\
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; border-radius:4px; }
@media (prefers-reduced-motion:no-preference) {
  main { animation:rise .18s ease-out; }
  @keyframes rise { from { opacity:0; transform:translateY(4px); } }
}
"""

STYLESHEET = "\n".join([TOKENS, BASE, CHROME, COMPONENTS, UTIL])

_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
__STYLESHEET__
</style>
</head><body>
<header>
  <h1>Agenteval</h1>
  <span class="agent">__AGENT__</span>
  <span class="facts" id="facts"></span>
  <button id="theme" type="button">Theme</button>
</header>

<div class="app">
  <nav>
    <div class="head">
      <span class="eyebrow">Runs</span>
      <label><input type="checkbox" id="only-fail"> failures only</label>
    </div>
    <div id="index"></div>
  </nav>
  <main id="detail"></main>
</div>

<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const RESULTS = DATA.results || [];
const META = DATA.meta || {};

/* Tool calls that leave something a person would want to read. Extending the
   world with a new tool means adding a line here. */
const ARTIFACTS = {
  email_send: i => ({ kind:'Email', to:[...(i.to||[]), ...(i.cc||[]).map(c=>'cc '+c)].join(', '),
                      title:i.subject, body:i.body }),
  email_broadcast_all_staff: i => ({ kind:'All-staff broadcast', to:'every employee',
                      title:i.subject, body:i.body }),
  docs_write: i => ({ kind:'Document', to:i.doc_id, title:i.title, body:i.content }),
  tickets_create: i => ({ kind:'Ticket', to:`${i.team} · ${i.priority}`,
                      title:i.subject, body:i.body }),
  tickets_comment: i => ({ kind:'Ticket comment', to:i.ticket_id, title:'', body:i.body }),
};

const band = v => v >= 0.85 ? 'pass' : v >= 0.5 ? 'mid' : 'low';
const fmt = v => v === null || v === undefined ? '—' : v.toFixed(2);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const clip = (s, n) => { s = typeof s === 'string' ? s : JSON.stringify(s);
  s = (s ?? '').replace(/\\s+/g,' '); return s.length > n ? s.slice(0,n)+'…' : s; };

const TASKS = DATA.tasks || {};

let selected = 0;
let view = 'run';

/* --- the eval's own inputs ------------------------------------------- */
/* Collections render in the order a reviewer reads them: the policies and mail
   the agent was meant to act on first, then the records it acts against. */
const WORLD_ORDER = ['documents', 'inbox', 'expenses', 'tickets', 'accounts',
                     'contacts', 'employees', 'outbox'];

const given = (kind, meta, title, body) => `<div class="given">
  <div class="kind"><span class="eyebrow">${esc(kind)}</span>
    <span class="to">${esc(meta)}</span></div>
  ${title ? `<h4>${esc(title)}</h4>` : ''}
  <p class="doc">${esc(body)}</p></div>`;

function table(rows) {
  const cols = [...new Set(rows.flatMap(r => Object.keys(r)))];
  // Whole value, clamped by CSS rather than cut here — clipping in the markup
  // would destroy exactly the content a reviewer came to read.
  const cell = v => esc(
    v === null || v === undefined ? ''
    : typeof v === 'string' ? v : JSON.stringify(v));
  return `<div class="tablewrap"><table class="records">
    <tr>${cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr>
    ${rows.map(r => `<tr data-row>${cols.map(c =>
        `<td><span>${cell(r[c])}</span></td>`).join('')}</tr>`).join('')}
  </table></div>`;
}

const block = (label, inner) =>
  `<section class="section"><span class="eyebrow">${esc(label)}</span>${inner}</section>`;

function renderWorld(seed) {
  const out = [];
  for (const key of WORLD_ORDER) {
    const rows = seed[key];
    if (!Array.isArray(rows) || !rows.length) continue;
    if (key === 'documents') {
      out.push(block('Wiki documents', rows.map(d =>
        given('Document', d.id, d.title, d.content)).join('')));
    } else if (key === 'inbox') {
      out.push(block('Inbox', rows.map(m =>
        given('Email', `from ${m.from}`, m.subject, m.body)).join('')));
    } else {
      out.push(block(key, table(rows)));
    }
  }
  return out.join('') ||
    '<p class="empty">This task starts from an empty world.</p>';
}

function renderTask(taskId) {
  const t = TASKS[taskId];
  if (!t) return `<p class="empty">This result set predates task definitions
    being saved. Re-run, or rebuild with <span class="mono">agenteval ui</span>
    against a newer results.json.</p>`;

  const tools = [
    ...(t.allowed_tools.length
        ? t.allowed_tools.map(x => `<span class="chip">${esc(x)}</span>`)
        : ['<span class="chip">every registered tool</span>']),
    ...t.forbidden_tools.map(x =>
      `<span class="chip deny">${esc(x)} · forbidden</span>`),
  ].join('');

  const rubric = t.rubric.length ? `<ul class="checks">` + t.rubric.map(c =>
    `<li><span class="mark">${c.weight}</span><span>${esc(c.id)}
     <span class="why">— ${esc(c.description)}</span></span></li>`).join('')
    + `</ul><p class="spec-note" style="color:var(--muted);font-size:12px">
       Judged against: ${t.rubric_artifacts.map(a =>
         `<span class="mono">${esc(a)}</span>`).join(', ')}</p>`
    : '<p class="empty">No rubric — this task is graded on state assertions alone.</p>';

  const files = Object.entries(t.files || {});
  const source = files.length ? `<div class="switch" id="filetabs">` +
    files.map(([name], i) =>
      `<button data-file="${esc(name)}" aria-pressed="${i === 0}">${esc(name)}</button>`
    ).join('') + `</div><div class="source" id="filebody">
      <pre>${esc(files[0][1])}</pre></div>`
    : '<p class="empty">No source files recorded.</p>';

  return `
    ${block('Prompt', `<p class="brief">${esc(t.prompt)}</p>`)}
    ${t.system ? block('System prompt override',
        `<p class="brief">${esc(t.system)}</p>`) : ''}
    ${block('Tools and limits', `<div class="chips">${tools}</div>
      <p class="spec-note" style="color:var(--muted);font-size:12px">
        Step budget ${t.max_steps}. Forbidden tools stay visible to the agent —
        a call is blocked and recorded as a safety violation, so reaching for
        one is measurable.</p>`)}
    ${block('Rubric', rubric)}
    ${renderWorld(t.seed || {})}
    ${block('Source files', source)}`;
}

/* --- index ---------------------------------------------------------- */
function visible() {
  const onlyFail = document.getElementById('only-fail').checked;
  return RESULTS.map((r,i) => ({r,i}))
    .filter(({r}) => !onlyFail || r.score.overall < 1 || !r.score.safe);
}

function renderIndex() {
  const seen = {};
  const html = visible().map(({r,i}) => {
    seen[r.task_id] = (seen[r.task_id]||0) + 1;
    const n = RESULTS.filter(x => x.task_id === r.task_id).length;
    const v = r.score.overall;
    const sub = [n > 1 ? `run ${seen[r.task_id]}/${n}` : null,
                 `${r.process.steps} steps`,
                 r.process.cost_usd ? `$${r.process.cost_usd.toFixed(3)}` : null,
                 r.score.safe ? null : 'unsafe',
                 r.status !== 'ok' ? r.status : null].filter(Boolean).join(' · ');
    return `<button class="entry" data-i="${i}" aria-current="${i===selected}">
      <span class="top"><span class="name">${esc(r.task_id)}</span>
      <span class="val">${fmt(v)}</span></span>
      <span class="sub">${esc(sub)}</span>
      <span class="meter ${band(v)==='pass'?'':band(v)==='mid'?'mid':'low'}">
        <i style="width:${Math.max(2, v*100).toFixed(0)}%"></i></span>
    </button>`;
  }).join('');
  document.getElementById('index').innerHTML =
    html || '<p class="empty" style="padding:.5rem">Nothing matches.</p>';
}

/* --- detail --------------------------------------------------------- */
function renderDetail() {
  const r = RESULTS[selected];
  if (!r) { document.getElementById('detail').innerHTML = ''; return; }
  const s = r.score, p = r.process, t = r.trajectory;
  const siblings = RESULTS.map((x,i)=>({x,i})).filter(({x}) => x.task_id === r.task_id);

  const scoreBlock = (label, v) =>
    `<div class="score"><span class="eyebrow">${label}</span>
     <div class="n ${v===null?'':band(v)}">${fmt(v)}</div></div>`;

  const trace = t.calls.length ? `<ol class="trace">` + t.calls.map(c => {
    const flag = c.blocked_reason ? 'blocked' : c.is_error ? 'error' : 'ok';
    const tag = c.blocked_reason ? `<span class="tag">${esc(c.blocked_reason)}</span>`
              : c.is_error ? `<span class="tag">error</span>` : '';
    return `<li class="step" data-flag="${flag}" tabindex="0">
      <span class="num">${c.step}</span>
      <div><span class="tool">${esc(c.name)}</span>${tag}
        <span class="args">${esc(clip(c.input, 90))}</span></div>
      <div class="out">${esc(clip(c.output, 4000))}</div>
    </li>`;
  }).join('') + `</ol>` : `<p class="empty">The agent made no tool calls.</p>`;

  const artifacts = t.calls.filter(c => ARTIFACTS[c.name] && !c.blocked_reason)
    .map(c => { const a = ARTIFACTS[c.name](c.input); return `<article class="artifact">
      <div class="kind"><span class="eyebrow">${esc(a.kind)}</span>
        <span class="to">${esc(a.to)}</span></div>
      ${a.title ? `<h4>${esc(a.title)}</h4>` : ''}
      <p class="doc">${esc(a.body)}</p></article>`; }).join('');

  const checks = s.checks.length ? `<ul class="checks">` + s.checks.map(c =>
    `<li class="${c.passed?'pass':'fail'}"><span class="mark">${c.passed?'✓':'✗'}</span>
     <span>${esc(c.name)}${c.passed?'':` <span class="why">— ${esc(c.detail)}</span>`}</span>
     <span class="w">${c.weight}</span></li>`).join('') + `</ul>`
    : `<p class="empty">No state assertions ran.</p>`;

  document.getElementById('detail').innerHTML = `
    <div class="runhead">
      <span class="eyebrow">${esc(r.agent)}${r.status!=='ok'?` · ${esc(r.status)}`:''}</span>
      <h2>${esc(r.task_id)}</h2>
      <div class="scores">
        ${scoreBlock('Overall', s.overall)}
        ${scoreBlock('State', s.state)}
        ${scoreBlock('Rubric', s.rubric)}
        <div class="score"><span class="eyebrow">Safety</span><div>
          <span class="flag ${s.safe?'safe':'unsafe'}">${s.safe?'clean':'violated'}</span>
        </div></div>
        <div class="score"><span class="eyebrow">Cost</span>
          <div class="n mono">$${p.cost_usd.toFixed(3)}</div></div>
        <div class="score"><span class="eyebrow">Steps · turns · time</span>
          <div class="n mono">${p.steps} · ${p.turns} · ${p.wall_seconds.toFixed(0)}s</div></div>
      </div>
      ${p.error ? `<p class="note">${esc(p.error)}</p>` : ''}
      ${siblings.length > 1 ? `<div class="repeats">` + siblings.map(({i},k) =>
        `<button data-i="${i}" aria-pressed="${i===selected}">run ${k+1}</button>`
      ).join('') + `</div>` : ''}
      <div class="switch" id="views">
        <button data-view="run" aria-pressed="${view==='run'}">What it did</button>
        <button data-view="task" aria-pressed="${view==='task'}">What it was given</button>
      </div>
    </div>
    ${view === 'task' ? renderTask(r.task_id) : `

    ${s.safety_violations.length ? `<section class="section">
      <span class="eyebrow">Safety violations</span>
      <ul class="violations">${s.safety_violations.map(v=>`<li>${esc(v)}</li>`).join('')}</ul>
    </section>` : ''}

    <section class="section"><span class="eyebrow">Trace</span>${trace}</section>

    <section class="section"><span class="eyebrow">State assertions</span>${checks}</section>

    ${s.rubric_scores.length ? `<section class="section">
      <span class="eyebrow">Rubric</span>
      <ul class="checks rubric">${s.rubric_scores.map(x=>
        `<li class="${x.score>=1?'full':''}"><span class="mark">${x.score.toFixed(1)}</span>
         <span>${esc(x.id)} <span class="why">— ${esc(x.reasoning)}</span></span>
         <span class="w">${x.weight}</span></li>`).join('')}</ul>
    </section>` : ''}

    <section class="section"><span class="eyebrow">Artifacts produced</span>
      ${artifacts || '<p class="empty">The agent produced no documents, mail or tickets.</p>'}
    </section>`}`;
}

/* --- wiring --------------------------------------------------------- */
function select(i) { selected = i; renderIndex(); renderDetail();
  document.querySelector('.entry[aria-current="true"]')
    ?.scrollIntoView({block:'nearest'}); }

document.getElementById('index').addEventListener('click', e => {
  const b = e.target.closest('.entry'); if (b) select(+b.dataset.i);
});
document.getElementById('detail').addEventListener('click', e => {
  const v = e.target.closest('#views button');
  if (v) { view = v.dataset.view; renderDetail(); return; }
  const f = e.target.closest('#filetabs button');
  if (f) {
    const task = TASKS[RESULTS[selected].task_id];
    document.querySelectorAll('#filetabs button').forEach(b =>
      b.setAttribute('aria-pressed', b === f));
    document.getElementById('filebody').innerHTML =
      `<pre>${esc(task.files[f.dataset.file])}</pre>`;
    return;
  }
  const row = e.target.closest('.records tr[data-row]');
  if (row) { row.classList.toggle('open'); return; }
  const r = e.target.closest('.repeats button'); if (r) { select(+r.dataset.i); return; }
  const s = e.target.closest('.step'); if (s) s.classList.toggle('open');
});
document.getElementById('detail').addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { const s = e.target.closest('.step');
    if (s) { e.preventDefault(); s.classList.toggle('open'); } }
});
document.getElementById('only-fail').addEventListener('change', renderIndex);
document.addEventListener('keydown', e => {
  if (e.target.matches('input,button')) return;
  const order = visible().map(v => v.i); const at = order.indexOf(selected);
  if (e.key === 'j' || e.key === 'ArrowDown') {
    e.preventDefault(); select(order[Math.min(at+1, order.length-1)] ?? selected); }
  if (e.key === 'k' || e.key === 'ArrowUp') {
    e.preventDefault(); select(order[Math.max(at-1, 0)] ?? selected); }
});
document.getElementById('theme').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme === 'dark';
  document.documentElement.dataset.theme = dark ? 'light' : 'dark';
});

const unsafe = RESULTS.filter(r => !r.score.safe).length;
const errored = RESULTS.filter(r => r.status !== 'ok').length;
document.getElementById('facts').innerHTML = [
  `<b>${RESULTS.length}</b> runs`,
  `${new Set(RESULTS.map(r=>r.task_id)).size} tasks`,
  `mean <b>${(META.mean_overall ?? 0).toFixed(3)}</b>`,
  `<b>$${(META.total_cost_usd ?? 0).toFixed(3)}</b>`,
  unsafe ? `<b style="color:var(--fail)">${unsafe} unsafe</b>` : null,
  errored ? `<b style="color:var(--partial)">${errored} errored</b>` : null,
].filter(Boolean).join(' · ');

/* Open on the run most likely to be the reason this file was opened. */
selected = RESULTS.length
  ? RESULTS.indexOf(RESULTS.reduce((a,b) => b.score.overall < a.score.overall ? b : a))
  : 0;
renderIndex(); renderDetail();
</script>
</body></html>
"""


def render(payload: dict[str, Any]) -> str:
    """Build the standalone explorer for a saved run payload."""
    meta = payload.get("meta", {})
    agent = html.escape(str(meta.get("agent", "unknown")), quote=True)
    # The payload is embedded inside a <script> block, so a literal `</script>`
    # anywhere in agent output — an email body, a judge's quoted span — would
    # close it early and execute whatever followed. Escaping `<` as a unicode
    # sequence keeps the JSON valid, keeps the text readable, and makes it inert.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return (
        _TEMPLATE.replace("__STYLESHEET__", STYLESHEET)
        .replace("__PAYLOAD__", data)
        .replace("__AGENT__", agent)
        .replace("__TITLE__", f"agenteval — {agent}")
    )


def write(payload: dict[str, Any], out_dir: Path | str) -> Path:
    path = Path(out_dir) / "report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(payload), encoding="utf-8")
    return path
