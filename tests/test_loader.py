"""Task loading, and the errors a task author will actually hit.

Every failure here should name the task and say what is wrong with it. A
loader that raises `KeyError: 'prompt'` costs someone a debugging session for
a missing line of YAML.
"""

import pytest

from agenteval.tasks import TaskError, discover, filter_by_tag, load_task

TASK_YAML = """\
id: sample
tags: [demo, writing]
max_steps: 12
prompt: |
  Do the thing.
forbidden_tools: [admin_delete_record]
rubric:
  - id: tone
    weight: 2.0
    description: Is it polite?
rubric_artifacts: [outbox]
"""

VERIFY_PY = """\
from agenteval import checks

def verify(world, trajectory):
    return checks().add("ok", True).done()

def safety(world, trajectory):
    return ["a violation"]

GOLD = [{"say": "done"}]
"""


def write_task(root, name="sample", task_yaml=TASK_YAML, verify=VERIFY_PY, seed=None):
    directory = root / name
    directory.mkdir(parents=True)
    if task_yaml is not None:
        (directory / "task.yaml").write_text(task_yaml)
    if verify is not None:
        (directory / "verify.py").write_text(verify)
    if seed is not None:
        (directory / "seed.json").write_text(seed)
    return directory


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_a_complete_task_loads_every_field(tmp_path):
    directory = write_task(tmp_path, seed='{"today": "2026-01-01"}')
    task = load_task(directory)

    assert task.id == "sample"
    assert task.spec.prompt == "Do the thing."   # stripped
    assert task.spec.tags == ["demo", "writing"]
    assert task.spec.max_steps == 12
    assert task.spec.forbidden_tools == ["admin_delete_record"]
    assert task.spec.seed == {"today": "2026-01-01"}
    assert task.spec.source_dir == str(directory)

    [criterion] = task.spec.rubric
    assert (criterion.id, criterion.weight) == ("tone", 2.0)

    assert task.verify(None, None)[0].passed is True
    assert task.safety(None, None) == ["a violation"]
    assert task.gold == [{"say": "done"}]


def test_the_directory_name_is_the_default_id(tmp_path):
    directory = write_task(tmp_path, name="from_dirname",
                           task_yaml="prompt: do it\n")
    assert load_task(directory).id == "from_dirname"


def test_optional_pieces_may_be_absent(tmp_path):
    directory = write_task(
        tmp_path,
        task_yaml="prompt: do it\n",
        verify="def verify(world, trajectory):\n    return []\n",
    )
    task = load_task(directory)
    assert task.spec.seed == {}          # no seed.json
    assert task.spec.rubric == []
    assert task.safety is None           # no safety()
    assert task.gold is None             # no GOLD
    assert task.spec.max_steps == 40     # default


def test_defaults_do_not_leak_between_tasks(tmp_path):
    """Mutable defaults shared across TaskSpec instances would be a nasty,
    hard-to-trace bug."""
    write_task(tmp_path, name="a", task_yaml="prompt: a\n")
    write_task(tmp_path, name="b", task_yaml="prompt: b\n")
    first, second = discover(tmp_path)
    first.spec.tags.append("mutated")
    assert second.spec.tags == []


# --------------------------------------------------------------------------- #
# Authoring errors
# --------------------------------------------------------------------------- #


def test_a_missing_task_yaml_is_named(tmp_path):
    directory = write_task(tmp_path, task_yaml=None)
    with pytest.raises(TaskError, match="no task.yaml"):
        load_task(directory)


def test_a_missing_prompt_names_the_task(tmp_path):
    directory = write_task(tmp_path, task_yaml="id: sample\ntags: []\n")
    with pytest.raises(TaskError, match="task sample has no prompt"):
        load_task(directory)


def test_a_missing_verify_py_names_the_task(tmp_path):
    directory = write_task(tmp_path, verify=None)
    with pytest.raises(TaskError, match="task sample has no verify.py"):
        load_task(directory)


def test_a_verify_py_without_verify_names_the_task(tmp_path):
    directory = write_task(tmp_path, verify="def something_else():\n    pass\n")
    with pytest.raises(TaskError, match="verify.py defines no verify"):
        load_task(directory)


def test_a_rubric_with_nothing_to_grade_is_rejected(tmp_path):
    """Otherwise the judge scores against no evidence and the failure is
    invisible — it just returns low scores."""
    directory = write_task(
        tmp_path,
        task_yaml=(
            "prompt: p\nrubric:\n  - id: tone\n    description: polite?\n"
        ),
    )
    with pytest.raises(TaskError, match="rubric but no rubric_artifacts"):
        load_task(directory)


def test_an_empty_task_yaml_is_a_clean_error(tmp_path):
    directory = write_task(tmp_path, task_yaml="")
    with pytest.raises(TaskError, match="no prompt"):
        load_task(directory)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def test_discover_returns_tasks_in_a_stable_order(tmp_path):
    for name in ("c_task", "a_task", "b_task"):
        write_task(tmp_path, name=name, task_yaml=f"prompt: {name}\n")
    assert [t.id for t in discover(tmp_path)] == ["a_task", "b_task", "c_task"]


def test_discover_ignores_directories_without_a_task_yaml(tmp_path):
    write_task(tmp_path, name="real", task_yaml="prompt: p\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "scratch.md").write_text("nothing here")
    assert [t.id for t in discover(tmp_path)] == ["real"]


def test_only_filter_preserves_the_requested_order(tmp_path):
    for name in ("a", "b", "c"):
        write_task(tmp_path, name=name, task_yaml=f"prompt: {name}\n")
    assert [t.id for t in discover(tmp_path, only=["c", "a"])] == ["c", "a"]


def test_an_unknown_task_id_lists_what_is_available(tmp_path):
    write_task(tmp_path, name="real", task_yaml="prompt: p\n")
    with pytest.raises(TaskError, match=r"unknown task\(s\) \['ghost'\].*'real'"):
        discover(tmp_path, only=["ghost"])


def test_a_missing_task_directory_is_a_clean_error(tmp_path):
    with pytest.raises(TaskError, match="does not exist"):
        discover(tmp_path / "nowhere")


def test_filter_by_tag(tmp_path):
    write_task(tmp_path, name="a", task_yaml="prompt: a\ntags: [writing]\n")
    write_task(tmp_path, name="b", task_yaml="prompt: b\ntags: [routing]\n")
    tasks = discover(tmp_path)
    assert [t.id for t in filter_by_tag(tasks, "writing")] == ["a"]
    assert filter_by_tag(tasks, "absent") == []


def test_two_tasks_with_verify_modules_do_not_collide(tmp_path):
    """Each verify.py is loaded under a task-scoped module name; a shared one
    would mean the second task silently reusing the first one's checks."""
    write_task(
        tmp_path,
        name="first",
        task_yaml="prompt: p\n",
        verify="MARKER = 'first'\ndef verify(w, t):\n    return []\n",
    )
    write_task(
        tmp_path,
        name="second",
        task_yaml="prompt: p\n",
        verify="MARKER = 'second'\ndef verify(w, t):\n    return []\n",
    )
    first, second = discover(tmp_path)
    import sys

    assert sys.modules["agenteval_task_first"].MARKER == "first"
    assert sys.modules["agenteval_task_second"].MARKER == "second"
    assert first.verify is not second.verify


# --------------------------------------------------------------------------- #
# Manifest — the eval's own inputs, carried with its results
# --------------------------------------------------------------------------- #


def test_manifest_carries_the_definition_and_the_source_files(tmp_path):
    """A result set recording only what the agent did is unreviewable: you
    cannot tell whether a failure was the agent's or the task's without the
    prompt, the seeded world, and the assertions that judged it."""
    directory = write_task(tmp_path, seed='{"today": "2026-01-01"}')
    m = load_task(directory).manifest()

    assert m["id"] == "sample"
    assert m["prompt"] == "Do the thing."
    assert m["forbidden_tools"] == ["admin_delete_record"]
    assert m["max_steps"] == 12
    assert m["seed"] == {"today": "2026-01-01"}
    assert m["rubric"] == [
        {"id": "tone", "description": "Is it polite?", "weight": 2.0}
    ]
    assert m["rubric_artifacts"] == ["outbox"]
    assert m["has_gold"] is True
    assert set(m["files"]) == {"task.yaml", "seed.json", "verify.py"}
    assert "def verify" in m["files"]["verify.py"]


def test_manifest_survives_missing_optional_files(tmp_path):
    directory = write_task(
        tmp_path, task_yaml="prompt: p\n",
        verify="def verify(w, t):\n    return []\n",
    )
    m = load_task(directory).manifest()
    assert set(m["files"]) == {"task.yaml", "verify.py"}  # no seed.json
    assert m["seed"] == {} and m["has_gold"] is False


def test_every_shipped_task_manifests_its_own_files():
    from agenteval.tasks import DEFAULT_TASK_ROOT, SOURCE_FILES, discover

    for task in discover(DEFAULT_TASK_ROOT):
        m = task.manifest()
        assert set(m["files"]) == set(SOURCE_FILES), m["id"]
        assert m["prompt"] and m["seed"], m["id"]
