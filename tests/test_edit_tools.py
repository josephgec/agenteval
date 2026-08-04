"""Editing a file instead of rewriting it.

`exec_write_file` makes an agent reproduce a whole file to change one line,
which on a real repository is where most of the score goes. The tools here are
the alternative, and almost all of their value is in the two edits they
*refuse*: a snippet that is not there, and a snippet that is there more than
once.

The logic runs on the host — read the file out, edit, write it back — so most
of this needs no container. The end-to-end tests need the exec image and skip
without it.
"""

import pytest

from agenteval import TaskSpec, Trajectory, World
from agenteval.exec import EXEC_TOOLS, attach
from agenteval.exec import environment as env_mod
from agenteval.registry import REGISTRY, ToolSession

FILE = "/workspace/m.py"
SOURCE = '''def greet(name):
    return "hi " + name

def farewell(name):
    return "hi " + name
'''


class FakeEnvironment:
    """A container holding one file in memory."""

    def __init__(self, content=SOURCE, readable=True, writable=True):
        self.spec = env_mod.EnvironmentSpec()
        self.container = "fake"
        self.content = content
        self.readable = readable
        self.writable = writable
        self.commands = []

    def read_file(self, path, limit=200_000):
        if not self.readable:
            return env_mod.ExecResult(1, "", "no such file")
        return env_mod.ExecResult(0, self.content, "")

    def write_file(self, path, content):
        if not self.writable:
            return env_mod.ExecResult(1, "", "read-only file system")
        self.content = content
        return env_mod.ExecResult(0, "", "")

    def exec(self, command, timeout=None):
        self.commands.append(command)
        return env_mod.ExecResult(0, "", "")


@pytest.fixture
def session():
    """A tool session wired to a fake container."""
    def build(environment):
        world = World({})
        attach(world, environment)
        spec = TaskSpec(id="t", prompt="p", environment={"image": "x"})
        return ToolSession(world, spec, Trajectory("t", "a")), world

    return build


def _call(session, name, **arguments):
    return session.call(name, arguments)


# --------------------------------------------------------------------------- #
# What the edit tool refuses
# --------------------------------------------------------------------------- #


def test_a_snippet_that_is_not_there_is_an_error(session):
    """The most expensive failure available here is the silent no-op: the agent
    believes it made a change, every later step reasons from that belief, and
    the run ends with a confident summary of work that never happened."""
    environment = FakeEnvironment()
    talk, _ = session(environment)
    text, is_error = _call(talk, "exec_edit_file", path=FILE,
                           old_text="return 'nope'", new_text="x")
    assert is_error and "not in" in text
    assert environment.content == SOURCE  # and nothing was touched


def test_the_not_found_message_names_the_usual_cause(session):
    talk, _ = session(FakeEnvironment())
    text, _ = _call(talk, "exec_edit_file", path=FILE,
                    old_text="return 'nope'", new_text="x")
    assert "Whitespace and indentation" in text


def test_an_ambiguous_snippet_is_an_error(session):
    """Replacing "the first occurrence" silently edits the wrong one, which is
    worse than not editing at all because it corrupts something that worked."""
    environment = FakeEnvironment()
    talk, _ = session(environment)
    text, is_error = _call(talk, "exec_edit_file", path=FILE,
                           old_text='return "hi " + name', new_text="x")
    assert is_error and "appears 2 times" in text
    assert environment.content == SOURCE


def test_the_ambiguous_message_says_how_to_fix_it(session):
    talk, _ = session(FakeEnvironment())
    text, _ = _call(talk, "exec_edit_file", path=FILE,
                    old_text='return "hi " + name', new_text="x")
    assert "surrounding lines" in text


def test_an_unreadable_file_is_an_error(session):
    talk, _ = session(FakeEnvironment(readable=False))
    text, is_error = _call(talk, "exec_edit_file", path=FILE,
                           old_text="a", new_text="b")
    assert is_error and "could not read" in text


def test_a_write_that_fails_is_an_error(session):
    talk, _ = session(FakeEnvironment(writable=False))
    text, is_error = _call(talk, "exec_edit_file", path=FILE,
                           old_text="def greet(name):", new_text="def hello(name):")
    assert is_error and "could not write" in text


# --------------------------------------------------------------------------- #
# What it does
# --------------------------------------------------------------------------- #


def test_a_unique_snippet_is_replaced(session):
    environment = FakeEnvironment()
    talk, _ = session(environment)
    text, is_error = _call(
        talk, "exec_edit_file", path=FILE,
        old_text='def farewell(name):\n    return "hi " + name',
        new_text='def farewell(name):\n    return "bye " + name',
    )
    assert not is_error
    assert 'return "bye " + name' in environment.content
    # And only the intended one: greet is untouched.
    assert environment.content.count('return "hi " + name') == 1


def test_the_edit_reports_where_it_landed(session):
    talk, _ = session(FakeEnvironment())
    text, _ = _call(talk, "exec_edit_file", path=FILE,
                    old_text="def farewell(name):", new_text="def bye(name):")
    assert "line 4" in text


def test_an_empty_replacement_deletes(session):
    environment = FakeEnvironment()
    talk, _ = session(environment)
    _call(talk, "exec_edit_file", path=FILE,
          old_text='\ndef farewell(name):\n    return "hi " + name\n', new_text="")
    assert "farewell" not in environment.content


def test_the_edit_is_on_the_audit_trail(session):
    """Like every other tool here. An edit that did not go through the session
    would be invisible to the safety signal and the step budget."""
    talk, world = session(FakeEnvironment())
    _call(talk, "exec_edit_file", path=FILE,
          old_text="def greet(name):", new_text="def hello(name):")
    assert any(m.action == "edit_file" for m in world.mutations)


# --------------------------------------------------------------------------- #
# Reading part of a file
# --------------------------------------------------------------------------- #


def test_reading_the_whole_file_is_still_the_default(session):
    environment = FakeEnvironment()
    talk, _ = session(environment)
    text, _ = _call(talk, "exec_read_file", path=FILE)
    assert text == SOURCE
    assert not environment.commands  # read_file, not a shell command


def test_a_line_range_goes_through_sed(session):
    environment = FakeEnvironment()
    talk, _ = session(environment)
    _call(talk, "exec_read_file", path=FILE, start_line=4, line_count=2)
    assert "sed -n 4,5p" in environment.commands[0]


def test_a_start_with_no_count_reads_to_the_end(session):
    """And the `$` gets quoted on the way, which a bare interpolation would
    have let the shell expand into nothing."""
    environment = FakeEnvironment()
    talk, _ = session(environment)
    _call(talk, "exec_read_file", path=FILE, start_line=3)
    assert "'3,$p'" in environment.commands[0]


def test_read_output_carries_no_line_numbers(session):
    """Deliberate, and the reason the edit tool is usable at all: an agent
    edits by quoting back an exact snippet of what it read, and numbers woven
    into that text turn every edit into a de-numbering exercise."""
    talk, _ = session(FakeEnvironment())
    text, _ = _call(talk, "exec_read_file", path=FILE)
    assert not text.lstrip().startswith("1")
    assert text.splitlines()[0] == "def greet(name):"


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


class SearchEnvironment(FakeEnvironment):
    def __init__(self, output=""):
        super().__init__()
        self.output = output

    def exec(self, command, timeout=None):
        self.commands.append(command)
        return env_mod.ExecResult(0 if self.output else 1, self.output, "")


def test_search_returns_matches_with_line_numbers(session):
    """Numbers belong here, unlike in read output: this is for navigating to a
    place, not for copying text out of."""
    talk, _ = session(SearchEnvironment("/workspace/m.py:1:def greet(name):\n"))
    text, is_error = _call(talk, "exec_search", pattern="def ")
    assert not is_error and ":1:" in text


def test_no_matches_is_an_answer_not_an_error(session):
    """grep exits 1 when nothing matched. That is a useful answer, and the
    agent should get it as one rather than as a failure to recover from."""
    talk, _ = session(SearchEnvironment(""))
    text, is_error = _call(talk, "exec_search", pattern="nowhere")
    assert not is_error and "No matches" in text


def test_search_skips_binaries(session):
    """A checked-out repository is mostly build artefacts as far as grep is
    concerned."""
    talk, _ = session(SearchEnvironment("x"))
    environment = SearchEnvironment("x")
    talk, _ = session(environment)
    _call(talk, "exec_search", pattern="x")
    assert "grep -rnI" in environment.commands[0]


def test_search_caps_its_output(session):
    environment = SearchEnvironment("x")
    talk, _ = session(environment)
    _call(talk, "exec_search", pattern="x", max_results=5)
    assert "head -5" in environment.commands[0]


def test_a_pattern_with_shell_metacharacters_is_quoted(session):
    """Passed through a shell, so `$(...)` in a search term would otherwise
    execute rather than be searched for."""
    environment = SearchEnvironment("")
    talk, _ = session(environment)
    _call(talk, "exec_search", pattern="$(touch /tmp/pwned)")
    assert "'$(touch /tmp/pwned)'" in environment.commands[0]


# --------------------------------------------------------------------------- #
# Exposure
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["exec_edit_file", "exec_search"])
def test_the_new_tools_are_registered_like_any_other(name):
    assert name in REGISTRY and name in EXEC_TOOLS


def test_the_code_benchmarks_offer_them(monkeypatch):
    """SWE-bench is where this matters: rewriting a whole source file to change
    three lines is most of the difference between scaffolds."""
    from agenteval.benchmarks import HumanEvalBenchmark

    benchmark = HumanEvalBenchmark()
    benchmark._problems = {"HumanEval/0": {
        "task_id": "HumanEval/0", "prompt": "def f():\n", "entry_point": "f",
        "canonical_solution": "    pass\n", "test": "def check(c):\n    pass\n",
    }}
    allowed = benchmark.load("HumanEval/0").spec.allowed_tools
    assert "exec_edit_file" in allowed and "exec_search" in allowed


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


IMAGE = env_mod.DEFAULT_IMAGE
needs_image = pytest.mark.skipif(
    not (env_mod.available() and env_mod.image_present(IMAGE)),
    reason=f"{IMAGE} not built",
)


@needs_image
def test_editing_a_real_file_in_a_real_container():
    from agenteval.exec import Environment, EnvironmentSpec

    spec = EnvironmentSpec(files={FILE: SOURCE})
    with Environment(spec) as environment:
        world = World({})
        attach(world, environment)
        talk = ToolSession(
            world, TaskSpec(id="t", prompt="p", environment={"image": IMAGE}),
            Trajectory("t", "a"),
        )
        text, is_error = _call(
            talk, "exec_edit_file", path=FILE,
            old_text='def farewell(name):\n    return "hi " + name',
            new_text='def farewell(name):\n    return "bye " + name',
        )
        assert not is_error, text
        on_disk = environment.read_file(FILE).stdout
        assert 'return "bye " + name' in on_disk
        assert on_disk.count('return "hi " + name') == 1


@needs_image
def test_searching_a_real_container():
    from agenteval.exec import Environment, EnvironmentSpec

    with Environment(EnvironmentSpec(files={FILE: SOURCE})) as environment:
        world = World({})
        attach(world, environment)
        talk = ToolSession(
            world, TaskSpec(id="t", prompt="p", environment={"image": IMAGE}),
            Trajectory("t", "a"),
        )
        text, is_error = _call(talk, "exec_search", pattern="farewell",
                               path="/workspace")
        assert not is_error
        assert f"{FILE}:4:" in text
