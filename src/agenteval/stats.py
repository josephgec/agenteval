"""How much of a difference is a difference.

Twenty instances of HumanEval gives a pass@1 with a 95% interval about thirty
points wide. Two models at 30% and 45% on that sample are indistinguishable,
and a table that prints `0.30` beside `0.45` invites exactly the opposite
conclusion. Everything here exists to stop a number looking more meaningful
than it is.

Three ideas, and the middle one is the one that earns its keep.

*An interval on every mean.* Binary outcomes — a benchmark instance either
resolved or it did not — get a Wilson score interval, which behaves sensibly at
the edges where the normal approximation puts the lower bound of a perfect
score above 100%. Continuous scores, which is what the enterprise tasks
produce, get a bootstrap percentile interval instead: those are weighted blends
of checks, not proportions, and treating them as coin flips would be wrong.

*Paired comparison.* Two models over the same instances are not two independent
samples. Comparing their marginal means throws away the pairing and needs far
more data to see the same effect; comparing per-instance differences keeps it.
Instance difficulty is the dominant source of variance in these benchmarks, and
pairing cancels it exactly.

*How much data would settle it.* `sample_size_for` answers the question
`--limit` is really asking, so choosing 20 or 200 is a decision rather than a
guess.

No SciPy. These are a dozen lines each and the dependency is not worth it for a
harness whose only other requirements are an SDK and a YAML parser.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

#: 95%, two-sided. Named because it appears in three formulas below and a bare
#: 1.96 in the middle of an expression is unreadable.
Z95 = 1.959963984540054


@dataclass
class Interval:
    """A point estimate and the range the evidence actually supports."""

    point: float
    low: float
    high: float
    n: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def excludes_zero(self) -> bool:
        """True when the sign of the estimate is established."""
        return self.low > 0 or self.high < 0

    def render(self, places: int = 2) -> str:
        return (
            f"{self.point:.{places}f} "
            f"[{self.low:.{places}f}, {self.high:.{places}f}]"
        )


def is_binary(scores: list[float]) -> bool:
    """Pass/fail, as every real benchmark here reports."""
    return bool(scores) and all(s in (0.0, 1.0) for s in scores)


def wilson(successes: float, n: int, z: float = Z95) -> Interval:
    """Wilson score interval for a proportion.

    Used rather than the textbook normal approximation because that one breaks
    exactly where benchmark results live: at 20 for 20 it puts the lower bound
    at 100%, claiming certainty from twenty observations, and at 0 for 20 it
    dips below zero.
    """
    if n <= 0:
        return Interval(0.0, 0.0, 0.0, 0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return Interval(p, max(0.0, centre - spread), min(1.0, centre + spread), n)


def bootstrap(
    values: list[float], z: float = Z95, rounds: int = 4000, seed: int = 0
) -> Interval:
    """Percentile bootstrap interval for a mean.

    For the weighted, continuous scores the enterprise tasks produce. Seeded, so
    the same results file always reports the same interval — a confidence bound
    that flickers between runs of the *reporting* code is worse than none.
    """
    if not values:
        return Interval(0.0, 0.0, 0.0, 0)
    n = len(values)
    point = statistics.fmean(values)
    if n == 1:
        return Interval(point, 0.0, 1.0, 1)
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choices(values, k=n)) for _ in range(rounds)
    )
    tail = (1 - _confidence(z)) / 2
    low = means[max(0, int(tail * rounds))]
    high = means[min(rounds - 1, int((1 - tail) * rounds))]
    return Interval(point, low, high, n)


def _confidence(z: float) -> float:
    """Two-sided confidence level for a z score, via the normal CDF."""
    return math.erf(z / math.sqrt(2))


def interval(scores: list[float], z: float = Z95) -> Interval:
    """The right interval for whatever these scores are."""
    if not scores:
        return Interval(0.0, 0.0, 0.0, 0)
    if is_binary(scores):
        return wilson(sum(scores), len(scores), z)
    return bootstrap(scores, z)


# --------------------------------------------------------------------------- #
# Comparing two models
# --------------------------------------------------------------------------- #


#: Below this, a percentile bootstrap has no resolution — with two observations
#: there are three distinct resamples, and an interval computed from them will
#: happily exclude zero while establishing nothing. Two instances that both
#: improved is a fact about two instances, not a comparison.
MIN_PAIRS = 5


@dataclass
class Comparison:
    """What can be said about two models over the same instances."""

    left: str
    right: str
    #: Instances both attempted. The only ones that carry information.
    paired: int
    #: Instances one ran and the other did not, which are simply excluded.
    unpaired: int
    difference: Interval
    wins: int
    losses: int
    ties: int

    @property
    def decisive(self) -> bool:
        return self.paired >= MIN_PAIRS and self.difference.excludes_zero()

    def verdict(self) -> str:
        if not self.paired:
            return "no instances in common — nothing to compare"
        if self.paired < MIN_PAIRS:
            return (
                f"only {self.paired} shared instance"
                f"{'s' if self.paired != 1 else ''} — too few to compare, "
                f"whichever way they went"
            )
        ahead = self.right if self.difference.point > 0 else self.left
        if not self.decisive:
            return (
                f"no detectable difference over {self.paired} shared instances "
                f"— the interval spans zero, so the sign is not established"
            )
        return (
            f"{ahead} is ahead by {abs(self.difference.point):.2f} "
            f"over {self.paired} shared instances"
        )


def compare(
    left: dict[str, list[float]],
    right: dict[str, list[float]],
    left_label: str = "A",
    right_label: str = "B",
    z: float = Z95,
    seed: int = 0,
) -> Comparison:
    """Paired comparison of two models, task id to that task's scores.

    Pairing is the whole point. Instance difficulty dominates the variance in
    every benchmark here — some HumanEval problems are trivial and some are
    not — and comparing marginal means leaves that variance in, needing several
    times the data to see the same effect. Differencing per instance removes it
    exactly, because both models faced the same problem.
    """
    shared = sorted(set(left) & set(right))
    differences = [
        statistics.fmean(right[task]) - statistics.fmean(left[task])
        for task in shared
    ]
    return Comparison(
        left=left_label,
        right=right_label,
        paired=len(shared),
        unpaired=len(set(left) ^ set(right)),
        # Bootstrapped over the per-instance differences rather than assuming
        # they are normal: with twenty mostly-zero-or-one differences they are
        # emphatically not.
        difference=bootstrap(differences, z, seed=seed) if differences
        else Interval(0.0, 0.0, 0.0, 0),
        wins=sum(1 for d in differences if d > 1e-9),
        losses=sum(1 for d in differences if d < -1e-9),
        ties=sum(1 for d in differences if abs(d) <= 1e-9),
    )


def sample_size_for(delta: float, p: float = 0.5, z: float = Z95) -> int:
    """Roughly how many instances to detect a difference of `delta`.

    The question `--limit` is actually asking. Deliberately the simple
    normal-approximation formula for a difference of proportions at 80% power:
    it is the right order of magnitude, and a more careful number would imply a
    precision that the assumption of independent instances does not support
    anyway.
    """
    if delta <= 0:
        return 0
    power_z = 0.8416  # 80% power, one-sided
    variance = 2 * p * (1 - p)
    return max(1, math.ceil(variance * (z + power_z) ** 2 / (delta * delta)))
