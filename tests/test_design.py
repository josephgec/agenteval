"""The design-system bundle pushed to claude.ai/design.

The property that matters is single-sourcing. A design system that describes an
older version of the product is worse than none, because it is consulted
precisely when someone is deciding what the product *should* look like. These
specimens are composed from the same stylesheet the report renders with, and
these tests are what stop that from quietly becoming untrue.
"""

import re

import pytest

from agenteval import design, ui

SPECS = design.SPECIMENS
IDS = [s.path for s in SPECS]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("design")
    paths = design.build(out)
    return {p.relative_to(out).as_posix(): p.read_text() for p in paths}


def test_every_specimen_is_written(built):
    assert sorted(built) == sorted(s.path for s in SPECS)


def test_paths_and_names_are_unique():
    assert len({s.path for s in SPECS}) == len(SPECS)
    assert len({s.name for s in SPECS}) == len(SPECS)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_card_marker_is_the_first_line(spec, built):
    """The Design System pane builds its index from this line, so a stray
    doctype above it means the specimen never appears as a card."""
    first = built[spec.path].split("\n", 1)[0]
    assert re.fullmatch(r'<!-- @dsCard group="[^"]+" -->', first), first
    assert f'group="{spec.group}"' in first


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_specimens_carry_the_live_tokens(spec, built):
    """Not a copy of the palette — the palette itself.

    If this drifts, the design system starts describing a product that no
    longer exists, and every decision taken from it is taken from fiction.
    """
    assert ui.TOKENS.strip() in built[spec.path]


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_specimens_carry_the_live_component_css(spec, built):
    assert ui.COMPONENTS.strip() in built[spec.path]


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_specimens_omit_the_explorer_shell(spec, built):
    """CHROME is the two-pane layout of one page, not part of the library.
    Shipping it would invite someone to treat the app's scaffolding as a
    reusable component."""
    assert "grid-template-columns:270px" not in built[spec.path]


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_specimens_are_self_contained(spec, built):
    """They render inside a pane that will not fetch anything for them, and
    they have to keep rendering years from now."""
    page = built[spec.path]
    for external in ("http://", "https://", "src=", "@import", "<script"):
        assert external not in page


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_specimens_are_annotated(spec, built):
    """A swatch with no rationale is a picture. The note is what makes it a
    specification someone can decide against."""
    assert spec.notes, f"{spec.path} has no rationale"
    assert len(spec.notes) > 80
    assert spec.notes in built[spec.path]


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_prose_is_constrained_to_the_card_width(spec, built):
    """Otherwise the rationale runs off the edge of the card it is cropped to."""
    assert f"body {{ max-width:{spec.width}px; }}" in built[spec.path]


def test_components_show_their_states_not_a_happy_path():
    """A trace step is only interesting because of what a blocked call looks
    like; a run entry only because of the failing one."""
    trace = next(s for s in SPECS if s.path.endswith("trace-step.html"))
    for state in ('data-flag="ok"', 'data-flag="blocked"', 'data-flag="error"',
                  "step open"):
        assert state in trace.body

    entry = next(s for s in SPECS if s.path.endswith("run-entry.html"))
    for state in ('aria-current="true"', 'class="meter low"', 'class="meter mid"'):
        assert state in entry.body

    verdicts = next(s for s in SPECS if s.path.endswith("verdict-rows.html"))
    for state in ('class="pass"', 'class="fail"', "violations", "rubric"):
        assert state in verdicts.body


def test_both_themes_are_specified():
    """The report ships light and dark; a library that only documents one of
    them leaves the other undesigned."""
    assert any(s.dark for s in SPECS)
    assert any(not s.dark for s in SPECS)
    dark = next(s for s in SPECS if s.dark)
    assert 'data-theme="dark"' in design._page(dark)


def test_groups_are_a_small_stable_set():
    """The pane sections by this, so a typo silently creates a new group."""
    assert {s.group for s in SPECS} == {
        "Foundations", "Trace", "Scoring", "Artifacts", "Navigation"
    }


def test_build_is_idempotent(tmp_path):
    first = {p: p.read_text() for p in design.build(tmp_path)}
    second = {p: p.read_text() for p in design.build(tmp_path)}
    assert first == second


def test_a_token_change_reaches_the_specimens(tmp_path, monkeypatch):
    """The single-sourcing claim, demonstrated rather than asserted."""
    monkeypatch.setattr(ui, "TOKENS", ui.TOKENS.replace("#2F5FD0", "#FF00FF"))
    monkeypatch.setattr(design, "TOKENS", ui.TOKENS)
    page = (design.build(tmp_path)[0]).read_text()
    assert "#FF00FF" in page
