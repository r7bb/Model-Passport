"""Decide how to use each column of an arbitrary table, and clean tables the same way every time.

``infer_plan`` looks at the training data once and assigns every column a role (numeric,
categorical, or datetime) or a reason to drop it. ``TableCleaner`` is the first step of every
model pipeline: it applies that plan to any later table, so batches with missing columns, extra
columns, numbers stored as text, or unseen values are handled instead of crashing.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from model_passport.scanners.base import load_table
from model_passport.scanners.data_pii import column_tokens

PARSE_MIN_RATE = 0.95  # share of values that must parse for a text column to change role
SAMPLE_ROWS = 2_000  # values inspected per column when inferring roles
FREE_TEXT_MIN_UNIQUE = 100
FREE_TEXT_UNIQUE_RATIO = 0.5
ID_UNIQUE_RATIO = 0.95
ID_NAME_TOKENS = {"id", "uuid", "guid", "index", "key", "identifier"}
DATETIME_PARTS = ("year", "month", "dayofweek", "days")
_LEADING_ZERO_RE = re.compile(r"^[+-]?0\d")
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


class Role(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"


@dataclass(frozen=True)
class FeaturePlan:
    """Role of every kept column, and why each other column was dropped."""

    roles: dict[str, Role]
    dropped: dict[str, str] = field(default_factory=dict)

    def columns(self, role: Role) -> list[str]:
        return [c for c, r in self.roles.items() if r is role]

    @property
    def numeric_outputs(self) -> list[str]:
        """Numeric columns after cleaning: numeric features plus expanded datetime parts."""
        expanded = [f"{c}__{part}" for c in self.columns(Role.DATETIME) for part in DATETIME_PARTS]
        return self.columns(Role.NUMERIC) + expanded

    @property
    def categorical_outputs(self) -> list[str]:
        return self.columns(Role.CATEGORICAL)

    def describe(self) -> dict[str, Any]:
        return {
            "roles": {c: r.value for c, r in self.roles.items()},
            "dropped": dict(self.dropped),
        }


def _numeric_text(values: pd.Series) -> pd.Series:
    """Parse text such as ``"1,234"`` or ``" 7.5 "``; unparseable values become NaN."""
    cleaned = values.astype(str).str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def numeric_values(values: pd.Series) -> pd.Series:
    """Numbers as floats, parsing text such as ``"1,234"``; anything else becomes NaN."""
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        return values.astype(float)
    return _numeric_text(values.where(values.notna(), "")).astype(float)


def datetime_days(values: pd.Series) -> pd.Series:
    """Days since 1970-01-01 for parseable dates; anything else becomes NaN."""
    return (_parse_datetimes(values) - _EPOCH).dt.days.astype(float)


def _parse_datetimes(values: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(values, errors="coerce", format="mixed", utc=True)


def _is_identifier(column: str, values: pd.Series) -> bool:
    """A row key: nearly every value distinct and a name like ``id`` or ``customer_id``."""
    unique_ratio = values.nunique() / len(values)
    named_like_id = bool(column_tokens(column) & ID_NAME_TOKENS) or column.startswith("Unnamed:")
    return named_like_id and unique_ratio >= ID_UNIQUE_RATIO and len(values) >= 20


def _text_role(values: pd.Series) -> Role | str:
    """Role for a text column, or the reason it cannot be used."""
    text = values.astype(str).str.strip()
    has_leading_zeros = bool(text.str.match(_LEADING_ZERO_RE).any())
    if not has_leading_zeros and _numeric_text(text).notna().mean() >= PARSE_MIN_RATE:
        return Role.NUMERIC
    # Bare digit strings (codes such as "01003") are never dates, whatever a parser accepts.
    looks_like_date = not text.str.fullmatch(r"\d+").all()
    if looks_like_date and _parse_datetimes(text).notna().mean() >= PARSE_MIN_RATE:
        return Role.DATETIME
    unique = values.nunique()
    if unique >= FREE_TEXT_MIN_UNIQUE and unique / len(values) >= FREE_TEXT_UNIQUE_RATIO:
        return "high-cardinality text (identifier or free text)"
    return Role.CATEGORICAL


def _drop_reason(column: str, values: pd.Series) -> str | None:
    if values.empty:
        return "all values missing"
    if values.nunique() == 1:
        return "constant"
    if _is_identifier(column, values):
        return "row identifier"
    return None


def column_role(column: str, series: pd.Series) -> Role | str:
    """Role of one column, or the reason it should be dropped."""
    values = series.dropna()
    reason = _drop_reason(column, values)
    if reason is not None:
        return reason
    if pd.api.types.is_bool_dtype(values):
        return Role.CATEGORICAL
    if pd.api.types.is_numeric_dtype(values):
        return Role.NUMERIC
    if pd.api.types.is_datetime64_any_dtype(values):
        return Role.DATETIME
    return _text_role(values)


def infer_plan(frame: pd.DataFrame, label: str | None, exclude: Iterable[str] = ()) -> FeaturePlan:
    """Assign every column except ``label`` and ``exclude`` a role or a reason to drop it."""
    skip = {*exclude, *([label] if label is not None else [])}
    sample = frame.sample(n=SAMPLE_ROWS, random_state=0) if len(frame) > SAMPLE_ROWS else frame
    roles: dict[str, Role] = {}
    dropped: dict[str, str] = {}
    for column in map(str, frame.columns):
        if column in skip:
            continue
        role = column_role(column, sample[column])
        if isinstance(role, Role):
            roles[column] = role
        else:
            dropped[column] = role
    return FeaturePlan(roles=roles, dropped=dropped)


def restore_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn text columns that are entirely numbers into numbers, except codes with leading zeros.

    Used after reading a CSV as text, so ``"02103"`` stays a code while ``"41"`` becomes 41.
    """
    out = frame.copy()
    for column in out.columns:
        series = out[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        values = series.dropna().astype(str).str.strip()
        if values.empty or values.str.match(_LEADING_ZERO_RE).any():
            continue
        parsed = pd.to_numeric(values, errors="coerce")
        if parsed.notna().all():
            numbers = pd.to_numeric(series, errors="coerce")
            integer = bool(np.all(np.mod(parsed.to_numpy(dtype=float), 1) == 0))
            complete = not numbers.isna().any()
            out[column] = numbers.astype("int64" if integer and complete else float)
    return out


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV, TSV, Parquet, or JSONL without letting type guessing corrupt codes."""
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return restore_types(pd.read_csv(path, sep=sep, dtype=str, keep_default_na=True))
    return load_table(path)


def category_text(value: Any) -> Any:
    """One text form per category: ``3``, ``3.0``, and ``"3"`` all become ``"3"``; NaN stays."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return np.nan
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


class TableCleaner(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Apply a ``FeaturePlan`` to any table: align columns, coerce types, expand datetimes.

    Missing columns become all-missing (the imputers downstream fill them), extra columns are
    ignored, and unparseable values become missing instead of raising.
    """

    def __init__(self, plan: FeaturePlan) -> None:
        self.plan = plan

    def fit(self, X: pd.DataFrame, y: Any = None) -> TableCleaner:  # noqa: ARG002 - sklearn API
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        parts: dict[str, pd.Series] = {}
        for column, role in self.plan.roles.items():
            series = frame[column] if column in frame.columns else pd.Series(np.nan, frame.index)
            if role is Role.NUMERIC:
                parts[column] = self._numeric(series)
            elif role is Role.CATEGORICAL:
                parts[column] = series.map(category_text).astype(object)
            else:
                parts.update(self._datetime(column, series))
        numeric = self.plan.numeric_outputs
        ordered = numeric + self.plan.categorical_outputs
        out = pd.DataFrame(parts, index=frame.index)
        return out.reindex(columns=ordered).astype(dict.fromkeys(numeric, float))

    @staticmethod
    def _numeric(series: pd.Series) -> pd.Series:
        return numeric_values(series)

    @staticmethod
    def _datetime(column: str, series: pd.Series) -> dict[str, pd.Series]:
        parsed = _parse_datetimes(series)
        return {
            f"{column}__year": parsed.dt.year.astype(float),
            f"{column}__month": parsed.dt.month.astype(float),
            f"{column}__dayofweek": parsed.dt.dayofweek.astype(float),
            f"{column}__days": (parsed - _EPOCH).dt.days.astype(float),
        }

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:  # noqa: ARG002
        return np.asarray(self.plan.numeric_outputs + self.plan.categorical_outputs, dtype=object)
