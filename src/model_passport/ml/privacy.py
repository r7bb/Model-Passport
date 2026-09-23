"""Privacy preprocessing for any table: drop direct identifiers, generalize quasi-identifiers.

The rules are learned once from the training data (``fit_generalization``) and stored as plain
JSON, then applied row by row (``apply_generalization``), so every later batch is transformed
exactly like the data the model was trained on.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from model_passport.scanners.base import ScanTarget
from model_passport.scanners.data_pii import PiiScanner
from model_passport.scanners.reid_risk import suggest_quasi_identifiers

MIN_BUCKETS = 3  # a numeric quasi-identifier is widened to at least this many ranges
TAIL_QUANTILE = 0.05  # values beyond the 5th and 95th percentiles share one open-ended bucket
PREFIX_MIN_LENGTH = 4
PREFIX_MASKED = 2
MIN_RANGE_VALUES = 10  # numeric columns with fewer distinct values are left as categories


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
    width = _nice_width(float(high_q - low_q), width)
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


def fit_generalization(
    frame: pd.DataFrame, columns: Iterable[str], width: float = 10
) -> dict[str, dict[str, Any]]:
    """Rules per column: numeric ranges, masked code suffixes, or none for categories."""
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
    return out
