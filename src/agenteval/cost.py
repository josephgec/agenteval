"""Per-run cost accounting.

Rates are USD per million tokens, first-party Claude API. Bedrock and Vertex
are partner-operated with their own pricing; if you route there, override
`PRICING` rather than trusting these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .types import Usage

#: Cache reads bill at ~0.1x input; 5-minute cache writes at ~1.25x input.
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class Rate:
    input_per_mtok: float
    output_per_mtok: float
    #: Promotional rate and the last day it applies, when one is running.
    intro_input_per_mtok: float | None = None
    intro_output_per_mtok: float | None = None
    intro_through: date | None = None

    def rates_on(self, on: date) -> tuple[float, float]:
        if (
            self.intro_through
            and on <= self.intro_through
            and self.intro_input_per_mtok is not None
            and self.intro_output_per_mtok is not None
        ):
            return self.intro_input_per_mtok, self.intro_output_per_mtok
        return self.input_per_mtok, self.output_per_mtok


PRICING: dict[str, Rate] = {
    "claude-fable-5": Rate(10.00, 50.00),
    "claude-mythos-5": Rate(10.00, 50.00),
    "claude-opus-5": Rate(5.00, 25.00),
    "claude-opus-4-8": Rate(5.00, 25.00),
    "claude-opus-4-7": Rate(5.00, 25.00),
    "claude-opus-4-6": Rate(5.00, 25.00),
    "claude-sonnet-5": Rate(
        3.00,
        15.00,
        intro_input_per_mtok=2.00,
        intro_output_per_mtok=10.00,
        intro_through=date(2026, 8, 31),
    ),
    "claude-sonnet-4-6": Rate(3.00, 15.00),
    "claude-haiku-4-5": Rate(1.00, 5.00),
}


class UnknownModel(Exception):
    """Raised rather than silently costing an unpriced model at zero."""


def cost_usd(model: str | None, usage: Usage, on: date | None = None) -> float:
    """Dollar cost of one run's token usage."""
    if model is None:  # e.g. the scripted agent — no model, no cost
        return 0.0
    rate = PRICING.get(model)
    if rate is None:
        raise UnknownModel(
            f"no price for {model!r}. Add it to agenteval.cost.PRICING — "
            "guessing would quietly under-report spend."
        )
    in_rate, out_rate = rate.rates_on(on or date.today())
    per_token_in = in_rate / 1_000_000
    per_token_out = out_rate / 1_000_000
    return (
        usage.input_tokens * per_token_in
        + usage.output_tokens * per_token_out
        + usage.cache_read_input_tokens * per_token_in * CACHE_READ_MULTIPLIER
        + usage.cache_creation_input_tokens * per_token_in * CACHE_WRITE_MULTIPLIER
    )
