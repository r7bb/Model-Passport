"""Independent confirmation of high and critical findings (the verification step).

Screening every entity finds candidates. A few can pass the false-discovery threshold by
chance, so each High or Critical finding is re-tested with fresh evidence before it can block
a release:

- **Models with log-probs:** the exposure test (Carlini et al., "The Secret Sharer", USENIX
  Security 2019). The real value is ranked against N fresh same-type alternatives, drawn from
  the same source as the screening references but independently, by the likelihood of the
  value and what follows it. A memorized value ranks at or near the top; a chance finding lands
  anywhere. The p-value is rank / (N + 1), and exposure is log2(N + 1) - log2(rank) bits.
- **Generation-only models:** the value is probed again with more samples, and the hits are
  tested against the pooled rate at which the model emits same-type controls.

Confirmation p-values are adjusted with Benjamini-Hochberg across the re-tested findings.
Findings that are not confirmed are capped at Low.
"""

from __future__ import annotations

import math

from scipy.stats import binom

from model_passport.llm.backends import GenerationModel, ScoringModel
from model_passport.llm.methods import Parts, Sample


def exposure_rank(
    model: ScoringModel, sample: Sample, references: list[str], window: int | None = None
) -> tuple[int, int]:
    """(rank of the real value among itself and ``references``, number of references)."""
    variants = [sample, *(sample.with_value(r) for r in references)]
    scored = model.score([v.text for v in variants])
    totals = []
    for result, variant in zip(scored, variants, strict=True):
        parts = Parts.of(result, variant.span, window)
        totals.append(float(parts.value.sum() + parts.suffix.sum()))
    rank = 1 + sum(total >= totals[0] for total in totals[1:])  # ties count against the value
    return rank, len(references)


def exposure_bits(rank: int, references: int) -> float:
    return math.log2(references + 1) - math.log2(rank)


def reprobe_p(
    model: GenerationModel,
    prompt: str,
    value: str,
    samples: int,
    control_rate: float,
    max_tokens: int,
) -> tuple[float, float]:
    """(p-value, extraction rate) of reproducing ``value`` in ``samples`` fresh completions."""
    outputs = model.complete(prompt, n=samples, max_tokens=max_tokens)
    wanted = " ".join(value.lower().split())
    hits = sum(wanted in " ".join(o.lower().split()) for o in outputs)
    return float(binom.sf(hits - 1, samples, control_rate)), hits / max(samples, 1)
