"""Membership scores for one sensitive value in its sentence (higher = more likely trained on).

A sample is a training sentence split around the value: ``prefix + value + suffix``. The
methods are those benchmarked by EL-MIA (Satvaty, Verberne, and Turkmen, LREC 2026):

- ``loss``: mean token log-likelihood of the whole sentence (Carlini et al., 2021).
- ``zlib``: total log-likelihood relative to the sentence's zlib-compressed size, so text that
  is merely easy to predict does not look memorized (Carlini et al., 2021).
- ``min_k``: mean of the k% least likely tokens (Min-K% Prob; Shi et al., 2024).
- ``recall``: how much the likelihood changes when a non-member prefix is prepended (ReCaLL;
  Xie et al., 2024).
- ``loss_suffix``: mean log-likelihood of only the tokens after the value.
- ``reference_set``: log-likelihood of the value minus the log mean likelihood of same-type
  alternatives in the same slot (EL-MIA's proposed method).
- ``reference_set_suffix``: the same, scoring only the continuation after the value, which
  removes the noise of the generic context (EL-MIA's best method).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from model_passport.llm.backends import Scored, logmeanexp

LIKELIHOOD_METHODS = (
    "loss",
    "zlib",
    "min_k",
    "recall",
    "loss_suffix",
    "reference_set",
    "reference_set_suffix",
)
PRIMARY_METHOD = "reference_set_suffix"
MIN_K_FRACTION = 0.2


@dataclass(frozen=True)
class Sample:
    prefix: str
    value: str
    suffix: str

    @property
    def text(self) -> str:
        return self.prefix + self.value + self.suffix

    @property
    def span(self) -> tuple[int, int]:
        return len(self.prefix), len(self.prefix) + len(self.value)

    def with_value(self, value: str) -> Sample:
        return Sample(self.prefix, value, self.suffix)


@dataclass(frozen=True)
class Parts:
    """Token log-probabilities of a sample, split into the value and the continuation."""

    all: np.ndarray
    value: np.ndarray
    suffix: np.ndarray

    @classmethod
    def of(
        cls, scored: Scored, span: tuple[int, int], window: int | None = None, text_start: int = 0
    ) -> Parts:
        """Split by character span; tokens before ``text_start`` (a prepended context) are
        ignored."""
        start, end = span
        every: list[float] = []
        value: list[float] = []
        suffix: list[float] = []
        for (token_start, token_end), lp in zip(scored.offsets, scored.logprobs, strict=True):
            if np.isnan(lp) or token_end <= text_start:
                continue
            every.append(lp)
            if token_start < end and token_end > start:
                value.append(lp)
            elif token_start >= end and (window is None or len(suffix) < window):
                suffix.append(lp)
        return cls(np.asarray(every), np.asarray(value), np.asarray(suffix))

    @property
    def continuation(self) -> np.ndarray:
        """The suffix, or the value itself when nothing follows it."""
        return self.suffix if self.suffix.size else self.value


def loss(parts: Parts) -> float:
    return float(parts.all.mean()) if parts.all.size else float("nan")


def zlib_ratio(parts: Parts, text: str) -> float:
    bits = 8 * len(zlib.compress(text.encode("utf-8")))
    return float(parts.all.sum()) / bits if bits else float("nan")


def min_k(parts: Parts, fraction: float = MIN_K_FRACTION) -> float:
    if not parts.all.size:
        return float("nan")
    count = max(1, round(fraction * parts.all.size))
    return float(np.sort(parts.all)[:count].mean())


def recall(unconditional: Parts, conditional: Parts) -> float:
    """ReCaLL ratio LL(x | non-member prefix) / LL(x); higher suggests membership."""
    base = loss(unconditional)
    return loss(conditional) / base if base else float("nan")


def loss_suffix(parts: Parts) -> float:
    return float(parts.continuation.mean()) if parts.continuation.size else float("nan")


def reference_set(candidate: Parts, references: list[Parts], suffix: bool) -> float:
    """log P(value | context) - log mean_i P(reference_i | context), per the EL-MIA LLR."""

    def total(parts: Parts) -> float:
        tokens = parts.continuation if suffix else parts.value
        return float(tokens.sum())

    if not references:
        return float("nan")
    return total(candidate) - logmeanexp([total(r) for r in references])
