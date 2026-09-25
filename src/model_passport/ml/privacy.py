"""Privacy preprocessing for any table: drop direct identifiers, generalize quasi-identifiers.

The rules are learned once from the training data (``fit_generalization``) and stored as plain
JSON, then applied row by row (``apply_generalization``), so every later batch is transformed
exactly like the data the model was trained on.

With a ``target_k``, generalization adapts to the data. Each column has a ladder of ever
coarser rules (wider numeric ranges, a shorter code prefix, or rare categories pooled into
"other"), ending with the column blanked. Starting from sensible buckets, each step moves one
column down its ladder, as many rungs as pays best: the move that removes the most at-risk
records (in groups smaller than ``target_k``) per unit of information lost about the label
(mutual information). Looking down the whole ladder avoids moves that only pay off several
rungs later being ignored. Columns that only identify people are coarsened first, and columns
that predict the label keep their detail. Steps stop once at most ``max_suppression`` of the
records are at risk; ``suppress_small_groups`` then removes those few. Generalization plus
suppression is the standard way to reach k-anonymity while keeping as much detail as possible.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from model_passport.scanners.base import ScanTarget
from model_passport.scanners.data_pii import PiiScanner
from model_passport.scanners.reid_risk import suggest_quasi_identifiers

MIN_BUCKETS = 3  # a numeric quasi-identifier is widened to at least this many ranges
TAIL_QUANTILE = 0.05  # values beyond the 5th and 95th percentiles share one open-ended bucket
PREFIX_MIN_LENGTH = 4
PREFIX_MASKED = 2
MIN_RANGE_VALUES = 10  # numeric columns with fewer distinct values are left as categories
MAX_COARSENING_STEPS = 40
LABEL_BINS = 5  # a numeric label is binned into quantiles to measure information about it
LOSS_FLOOR = 1e-3  # keeps the gain-per-loss ratio finite for steps that lose nothing
OTHER = "other"
SUPPRESSED = "*"


def detect_identifiers(frame: pd.DataFrame, keep: Iterable[str] = ()) -> list[str]:
    """Columns holding direct identifiers (names, emails, phones, IDs), found by the PII scan."""
    findings = PiiScanner().scan(ScanTarget.from_frame(frame, name="input"))
    protected = set(keep)
    columns = {
        f.location.split(":", 1)[1]
        for f in findings
        if f.details.get("direct_identifier") and f.location and ":" in f.location
    }
    return [str(c) for c in frame.columns if str(c) in columns - protected]


def quasi_identifiers(
    frame: pd.DataFrame, requested: Iterable[str] | str | None, exclude: Iterable[str] = ()
) -> list[str]:
    """``requested`` columns that exist, or suggestions from column names when "auto"."""
    skip = set(exclude)
    if requested in (None, "auto"):
        columns = suggest_quasi_identifiers(frame)
    elif isinstance(requested, str):
        columns = [requested]
    else:
        columns = list(requested)
    return [c for c in columns if c in frame.columns and c not in skip]


def _nice_width(span: float, width: float) -> float:
    """``width``, or the largest round number (1, 2, 5 x 10^k) giving ``MIN_BUCKETS`` ranges."""
    if span <= 0 or span / width >= MIN_BUCKETS:
        return width
    raw = span / MIN_BUCKETS
    magnitude = 10 ** math.floor(math.log10(raw))
    return float(next(m * magnitude for m in (5, 2, 1) if m * magnitude <= raw))


def _bucket_start(value: float, width: float) -> float:
    # The epsilon keeps 0.6 / 0.2 (= 2.9999...) in the bucket that starts at 0.6.
    return round(math.floor(value / width + 1e-9) * width, 10)


def _numeric_rule(values: pd.Series, width: float) -> dict[str, Any]:
    low_q, high_q = values.quantile([TAIL_QUANTILE, 1 - TAIL_QUANTILE])
    return _range_rule(values, _nice_width(float(high_q - low_q), width))


def _range_rule(values: pd.Series, width: float) -> dict[str, Any]:
    low_q, high_q = values.quantile([TAIL_QUANTILE, 1 - TAIL_QUANTILE])
    low = _bucket_start(low_q, width)
    high = max(_bucket_start(high_q, width), low + width)
    integer = bool(np.all(np.mod(values.to_numpy(dtype=float), 1) == 0))
    return {"kind": "range", "width": width, "low": low, "high": high, "integer": integer}


def _is_code(values: pd.Series) -> bool:
    """Fixed-length digit strings such as ZIP or postal codes."""
    text = values.astype(str)
    lengths = text.str.len()
    same_length = lengths.nunique() == 1 and lengths.iloc[0] >= PREFIX_MIN_LENGTH
    return bool(text.str.isdigit().all()) and bool(same_length)


def _initial_rules(frame: pd.DataFrame, columns: list[str], width: float) -> dict[str, Any]:
    rules: dict[str, dict[str, Any]] = {}
    for column in columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
        if numeric and values.nunique() >= MIN_RANGE_VALUES:
            rules[column] = _numeric_rule(values, width)
        elif _is_code(values):
            keep = len(str(values.iloc[0])) - PREFIX_MASKED
            rules[column] = {"kind": "prefix", "keep": keep}
    return rules


def _pooled(rule: dict[str, Any] | None, values: pd.Series) -> dict[str, Any] | None:
    counts = values.astype(str).value_counts()
    keep = [c for c in counts.index if rule is None or c in rule["keep"]]
    if len(keep) <= 2:
        return None
    return {"kind": "pool", "keep": sorted(keep[:-1])}  # the rarest kept category joins "other"


def _coarser(rule: dict[str, Any] | None, values: pd.Series) -> dict[str, Any] | None:
    """One step coarser than ``rule``; blanks the column once nothing coarser is left."""
    step = _generalize_further(rule, values)
    if step is None and (rule is None or rule["kind"] != "suppress"):
        return {"kind": "suppress"}
    return step


def _generalize_further(rule: dict[str, Any] | None, values: pd.Series) -> dict[str, Any] | None:
    if rule is None or rule["kind"] == "pool":
        return _pooled(rule, values)
    if rule["kind"] == "range":
        if rule["high"] <= rule["low"] + rule["width"]:
            return None  # already just two ranges
        return _range_rule(values, rule["width"] * 2)
    if rule["kind"] == "prefix":
        return {"kind": "prefix", "keep": rule["keep"] - 1} if rule["keep"] > 1 else None
    return None


def at_risk(frame: pd.DataFrame, columns: list[str], k: int) -> int:
    """Records in groups of fewer than ``k`` that share the same quasi-identifier values."""
    if not columns or frame.empty:
        return 0
    sizes = frame.groupby(columns, dropna=False, observed=True).size()
    return int(sizes[sizes < k].sum())


def _label_codes(label: pd.Series | None) -> pd.Series | None:
    if label is None:
        return None
    numeric = pd.api.types.is_numeric_dtype(label) and not pd.api.types.is_bool_dtype(label)
    if numeric and label.nunique() > LABEL_BINS:
        return pd.qcut(label.rank(method="first"), LABEL_BINS, labels=False)
    return label.astype(str)


def _information(column: pd.Series, label: pd.Series | None) -> float:
    """Mutual information with the label, or entropy of the column when there is no label."""
    values = column.astype(str)
    target = values if label is None else label
    return float(mutual_info_score(target, values))


@dataclass(frozen=True)
class _Step:
    column: str
    rule: dict[str, Any]
    gain: int  # at-risk records removed by this step
    loss: float  # information about the label lost by this step

    @property
    def value(self) -> tuple[float, float]:
        return (self.gain / (self.loss + LOSS_FLOOR), -self.loss)


def _ladder(rule: dict[str, Any] | None, values: pd.Series) -> list[dict[str, Any]]:
    """Every coarser rule for a column, one rung at a time, ending with the column blanked."""
    rungs = []
    for _ in range(MAX_COARSENING_STEPS):
        rule = _coarser(rule, values)
        if rule is None:
            break
        rungs.append(rule)
    return rungs


def _candidate_steps(
    frame: pd.DataFrame, names: list[str], rules: dict[str, Any], label: pd.Series | None, k: int
) -> list[_Step]:
    current = apply_generalization(frame, rules)
    risk = at_risk(current, names, k)
    steps = []
    for column in names:
        before = _information(current[column], label)
        for rule in _ladder(rules.get(column), frame[column].dropna()):
            trial = apply_generalization(frame, {**rules, column: rule})
            gain = risk - at_risk(trial, names, k)
            loss = max(before - _information(trial[column], label), 0.0)
            steps.append(_Step(column, rule, gain, loss))
    return steps


def fit_generalization(
    frame: pd.DataFrame,
    columns: Iterable[str],
    width: float = 10,
    target_k: int | None = None,
    max_suppression: float = 0.01,
    label: pd.Series | None = None,
) -> dict[str, dict[str, Any]]:
    """Rules per column: numeric ranges, masked code suffixes, pooled or blanked categories.

    Without ``target_k`` only the initial buckets are used. With it, rules are coarsened until
    at most ``max_suppression`` of the rows are in groups smaller than ``target_k``, choosing
    each step by at-risk records removed per unit of information lost about ``label``.
    """
    names = list(columns)
    rules = _initial_rules(frame, names, width)
    if not target_k or not names:
        return rules
    subset = frame[names]
    codes = _label_codes(label.loc[subset.index]) if label is not None else None
    allowed = max_suppression * len(frame)
    for _ in range(MAX_COARSENING_STEPS):
        if at_risk(apply_generalization(subset, rules), names, target_k) <= allowed:
            break
        steps = _candidate_steps(subset, names, rules, codes, target_k)
        if not steps:
            break
        best = max(steps, key=lambda step: step.value)
        rules = {**rules, best.column: best.rule}
    return rules


def suppress_small_groups(
    frame: pd.DataFrame, columns: list[str], k: int
) -> tuple[pd.DataFrame, int]:
    """Drop records in groups of fewer than ``k``; returns the rest and how many were dropped."""
    if not columns or frame.empty:
        return frame, 0
    sizes = frame.groupby(columns, dropna=False, observed=True)[columns[0]].transform("size")
    keep = sizes.fillna(0) >= k
    return frame[keep], int((~keep).sum())


def _format(value: float, integer: bool) -> str:
    return str(int(value)) if integer or float(value).is_integer() else f"{value:g}"


def range_label(value: Any, rule: dict[str, Any]) -> Any:
    """The bucket for ``value``: ``"<30"``, ``"30-39"``, or ``"70+"`` (missing stays missing)."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return np.nan
    width, low, high, integer = rule["width"], rule["low"], rule["high"], rule["integer"]
    start = _bucket_start(number, width)
    if start <= low:
        return f"<{_format(low + width, integer)}"
    if start >= high:
        return f"{_format(high, integer)}+"
    end = start + width - 1 if integer and float(width).is_integer() else start + width
    return f"{_format(start, integer)}-{_format(end, integer)}"


def prefix_label(value: Any, keep: int) -> Any:
    if pd.isna(value):
        return np.nan
    text = str(value)
    return text[:keep] + "*" * max(len(text) - keep, 0)


def pool_label(value: Any, keep: set[str]) -> Any:
    if pd.isna(value):
        return np.nan
    return str(value) if str(value) in keep else OTHER


def apply_generalization(frame: pd.DataFrame, rules: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Apply ``fit_generalization`` rules to any table with some or all of the columns."""
    out = frame.copy()
    for column, rule in rules.items():
        if column not in out.columns:
            continue
        if rule["kind"] == "range":
            out[column] = out[column].map(lambda v, r=rule: range_label(v, r)).astype(object)
        elif rule["kind"] == "prefix":
            out[column] = out[column].map(lambda v, k=rule["keep"]: prefix_label(v, k))
        elif rule["kind"] == "pool":
            out[column] = out[column].map(lambda v, keep=set(rule["keep"]): pool_label(v, keep))
        elif rule["kind"] == "suppress":
            out[column] = SUPPRESSED
    return out
