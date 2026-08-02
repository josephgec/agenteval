"""LLMJudge: verdict mapping, weighting, blindness, and failure handling.

No API calls. The behaviours that matter here are all about what the judge is
shown and what is done with what it returns.
"""

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from agenteval import TaskSpec, Trajectory, World
from agenteval.grading.judge import JudgeError, JudgeVerdict, LLMJudge
from agenteval.types import RubricCriterion

SEED = {
    "documents": [{"id": "policy/x", "title": "Policy", "content": "the rules"}],
    "tickets": [
        {"id": "TKT-1", "subject": "s", "body": "b", "status": "open",
         "priority": "P3", "team": "support", "comments": []}
    ],
}


class StubClient:
    """Returns a canned verdict and records the request it was sent."""

    def __init__(self, verdict=None, stop_reason="end_turn", raises=None,
                 input_tokens=1000, output_tokens=200):
        self._verdict = verdict
        self._stop_reason = stop_reason
        self._raises = raises
        self._usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        self.requests = []
        self.messages = self

    async def parse(self, **kwargs):
        self.requests.append(kwargs)
        if self._raises:
            raise self._raises
        return SimpleNamespace(
            stop_reason=self._stop_reason,
            parsed_output=self._verdict,
            usage=self._usage,
        )


def verdict(*pairs):
    return JudgeVerdict.model_validate(
        {"criteria": [{"id": i, "verdict": v, "reasoning": f"because {i}"}
                      for i, v in pairs]}
    )


def make_task(criteria, artifacts=("final_text",)):
    return TaskSpec(
        id="t",
        prompt="do the thing",
        seed=SEED,
        rubric=[RubricCriterion(id=i, description=d, weight=w)
                for i, d, w in criteria],
        rubric_artifacts=list(artifacts),
    )


def make_trajectory(final_text="I did the thing"):
    trajectory = Trajectory(task_id="t", agent="a")
    trajectory.final_text = final_text
    return trajectory


# --------------------------------------------------------------------------- #
# Verdict handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "label,expected", [("pass", 1.0), ("partial", 0.5), ("fail", 0.0)]
)
async def test_discrete_verdicts_map_to_scores(label, expected):
    task = make_task([("a", "criterion a", 1.0)])
    judge = LLMJudge(client=StubClient(verdict(("a", label))))
    [score] = (await judge.score(task, World(SEED), make_trajectory())).scores
    assert score.score == expected
    assert score.reasoning == "because a"


async def test_weights_come_from_the_task_not_the_judge():
    """The judge grades; the task decides what each criterion is worth."""
    task = make_task([("a", "cheap", 0.5), ("b", "expensive", 3.0)])
    judge = LLMJudge(client=StubClient(verdict(("a", "fail"), ("b", "pass"))))
    scores = (await judge.score(task, World(SEED), make_trajectory())).scores
    assert {s.id: s.weight for s in scores} == {"a": 0.5, "b": 3.0}


async def test_results_follow_the_task_order_not_the_response_order():
    task = make_task([("a", "first", 1.0), ("b", "second", 1.0)])
    # Judge answers out of order
    judge = LLMJudge(client=StubClient(verdict(("b", "pass"), ("a", "fail"))))
    scores = (await judge.score(task, World(SEED), make_trajectory())).scores
    assert [s.id for s in scores] == ["a", "b"]
    assert [s.score for s in scores] == [0.0, 1.0]


async def test_a_task_without_a_rubric_skips_the_api_entirely():
    """No rubric means no judging cost, and no accidental blank verdict."""
    client = StubClient(verdict())
    task = TaskSpec(id="t", prompt="p", seed=SEED)
    outcome = await LLMJudge(client=client).score(
        task, World(SEED), make_trajectory()
    )
    assert outcome.scores == []
    assert client.requests == []


# --------------------------------------------------------------------------- #
# What the judge is shown
# --------------------------------------------------------------------------- #


async def test_artifacts_named_by_the_task_reach_the_prompt():
    task = make_task([("a", "is the doc right", 1.0)], artifacts=["doc:policy/x"])
    client = StubClient(verdict(("a", "pass")))
    await LLMJudge(client=client).score(task, World(SEED), make_trajectory())

    prompt = client.requests[0]["messages"][0]["content"]
    assert "the rules" in prompt          # the artifact body
    assert "is the doc right" in prompt   # the criterion
    assert "do the thing" in prompt       # the original request


async def test_the_judge_is_not_shown_the_state_checks():
    """Anchoring on the programmatic result would stop the two signals from
    being independent."""
    task = make_task([("a", "criterion a", 1.0)])
    trajectory = make_trajectory()
    client = StubClient(verdict(("a", "pass")))
    await LLMJudge(client=client).score(task, World(SEED), trajectory)

    request = str(client.requests[0])
    for leak in ("state_check", "Check(", "passed=True", "state_score"):
        assert leak not in request


async def test_absent_artifacts_are_reported_rather_than_omitted():
    """Silence would read to the judge as 'nothing to grade' instead of
    'the agent produced nothing', which are different verdicts."""
    task = make_task([("a", "did they write it", 1.0)],
                     artifacts=["doc:postmortems/missing"])
    client = StubClient(verdict(("a", "fail")))
    await LLMJudge(client=client).score(task, World(SEED), make_trajectory())
    assert "no document exists at postmortems/missing" in (
        client.requests[0]["messages"][0]["content"]
    )


async def test_request_pins_model_effort_and_a_cache_breakpoint():
    task = make_task([("a", "a", 1.0)])
    client = StubClient(verdict(("a", "pass")))
    await LLMJudge(model="claude-opus-5", effort="medium", client=client).score(
        task, World(SEED), make_trajectory()
    )
    request = client.requests[0]
    assert request["model"] == "claude-opus-5"
    assert request["output_config"] == {"effort": "medium"}
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["output_format"] is JudgeVerdict


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #
# Every one of these must raise. A judge failure that returned [] instead would
# silently degrade the run to state-only scoring and make it look better than
# it was — see runner.run_one, which marks the run harness_error.


async def test_a_refusing_judge_raises():
    task = make_task([("a", "a", 1.0)])
    judge = LLMJudge(client=StubClient(verdict(), stop_reason="refusal"))
    with pytest.raises(JudgeError, match="refused"):
        await judge.score(task, World(SEED), make_trajectory())


async def test_an_unparseable_verdict_raises():
    task = make_task([("a", "a", 1.0)])
    judge = LLMJudge(client=StubClient(None))
    with pytest.raises(JudgeError, match="no parseable verdict"):
        await judge.score(task, World(SEED), make_trajectory())


async def test_a_partial_verdict_raises_naming_the_gap():
    """Scoring only the criteria that came back would quietly reweight the
    rubric toward whatever the judge happened to answer."""
    task = make_task([("a", "a", 1.0), ("b", "b", 1.0), ("c", "c", 1.0)])
    judge = LLMJudge(client=StubClient(verdict(("a", "pass"))))
    with pytest.raises(JudgeError, match=r"omitted criteria.*\['b', 'c'\]"):
        await judge.score(task, World(SEED), make_trajectory())


async def test_an_api_failure_is_wrapped_not_leaked():
    task = make_task([("a", "a", 1.0)])
    error = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    judge = LLMJudge(client=StubClient(raises=error))
    with pytest.raises(JudgeError, match="judge API call failed"):
        await judge.score(task, World(SEED), make_trajectory())


async def test_extra_criteria_from_the_judge_are_ignored():
    """A hallucinated criterion should not sneak weight into the score."""
    task = make_task([("a", "a", 1.0)])
    judge = LLMJudge(client=StubClient(verdict(("a", "pass"), ("invented", "pass"))))
    scores = (await judge.score(task, World(SEED), make_trajectory())).scores
    assert [s.id for s in scores] == ["a"]


# --------------------------------------------------------------------------- #
# Cost accounting
# --------------------------------------------------------------------------- #
# The judge is a second billed model. Attributing its spend to the run is what
# stops a locally-hosted agent graded by a hosted judge from reporting $0.00.


async def test_the_outcome_carries_usage_and_the_judging_model():
    task = make_task([("a", "a", 1.0)])
    client = StubClient(verdict(("a", "pass")), input_tokens=4000, output_tokens=300)
    outcome = await LLMJudge(model="claude-opus-5", client=client).score(
        task, World(SEED), make_trajectory()
    )
    assert outcome.usage.input_tokens == 4000
    assert outcome.usage.output_tokens == 300
    assert outcome.model == "claude-opus-5"


async def test_an_unjudged_task_reports_zero_usage():
    task = TaskSpec(id="t", prompt="p", seed=SEED)
    outcome = await LLMJudge(client=StubClient(verdict())).score(
        task, World(SEED), make_trajectory()
    )
    assert outcome.usage.input_tokens == 0
    assert outcome.usage.output_tokens == 0


@pytest.mark.parametrize(
    "client_kwargs,match",
    [
        ({"stop_reason": "refusal"}, "refused"),
        ({"verdict": None}, "no parseable verdict"),
    ],
)
async def test_a_failed_verdict_still_reports_what_it_cost(client_kwargs, match):
    """A refusal or a malformed verdict is a billed call. Dropping its usage on
    the error path is how cost reports drift low on exactly the runs you retry.
    """
    task = make_task([("a", "a", 1.0)])
    kwargs = {"verdict": verdict(("a", "pass")), **client_kwargs}
    judge = LLMJudge(client=StubClient(input_tokens=2500, output_tokens=80,
                                       **kwargs))
    with pytest.raises(JudgeError, match=match) as caught:
        await judge.score(task, World(SEED), make_trajectory())
    assert caught.value.usage.input_tokens == 2500
    assert caught.value.usage.output_tokens == 80


async def test_a_transport_failure_reports_no_usage():
    """Nothing came back, so nothing is known to have been billed."""
    task = make_task([("a", "a", 1.0)])
    error = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    judge = LLMJudge(client=StubClient(raises=error))
    with pytest.raises(JudgeError) as caught:
        await judge.score(task, World(SEED), make_trajectory())
    assert caught.value.usage.input_tokens == 0
