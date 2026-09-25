"""Detect the learning task from the label, and clean and split rows so training cannot break."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MAX_INTEGER_CLASSES = 10  # an integer label with more distinct values is treated as a quantity
MIN_CLASS_ROWS = 3  # smaller classes cannot appear in both splits and in every CV fold
MIN_ROWS = 10
REGRESSION_STRATA = 5
MIN_ROWS_PER_STRATUM = 10


class Task(StrEnum):
    BINARY = "binary-classification"
    MULTICLASS = "multiclass-classification"
    REGRESSION = "regression"

    @property
    def is_classification(self) -> bool:
        return self is not Task.REGRESSION


class DataError(ValueError):
    """The data cannot be used to train a model; the message says why and what to change."""


def infer_task(label: pd.Series, override: str | None = None) -> Task:
    """Binary, multiclass, or regression, from the label's type and distinct values."""
    values = label.dropna()
    if override not in (None, "", "auto"):
        return Task(override)
    distinct = values.nunique()
    if distinct < 2:
        raise DataError(f"label {label.name!r} has {distinct} distinct value(s); need at least 2")
    numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
    if numeric:
        integer_valued = bool(np.all(np.mod(values.to_numpy(dtype=float), 1) == 0))
        if not integer_valued or distinct > MAX_INTEGER_CLASSES:
            return Task.REGRESSION
    return Task.BINARY if distinct == 2 else Task.MULTICLASS


@dataclass
class RowReport:
    """What ``clean_rows`` removed, for the stage log and the preprocessing manifest."""

    rows_in: int
    missing_label: int = 0
    duplicates: int = 0
    rare_classes: dict[str, int] = field(default_factory=dict)

    @property
    def rows_out(self) -> int:
        rare = sum(self.rare_classes.values())
        return self.rows_in - self.missing_label - self.duplicates - rare

    def describe(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "missing_label": self.missing_label,
            "duplicates": self.duplicates,
            "rare_classes_dropped": dict(self.rare_classes),
        }


def clean_rows(
    frame: pd.DataFrame, label: str, task: Task, min_class_rows: int = MIN_CLASS_ROWS
) -> tuple[pd.DataFrame, RowReport]:
    """Drop rows with no label, exact duplicates, and classes too rare to learn or evaluate."""
    if label not in frame.columns:
        raise DataError(f"label column {label!r} not found; columns are {list(frame.columns)}")
    report = RowReport(rows_in=len(frame))
    out = frame.dropna(subset=[label])
    report.missing_label = len(frame) - len(out)
    deduped = out.drop_duplicates()
    report.duplicates = len(out) - len(deduped)
    out = deduped
    if task.is_classification:
        counts = out[label].value_counts()
        rare = counts[counts < min_class_rows]
        report.rare_classes = {str(k): int(v) for k, v in rare.items()}
        out = out[~out[label].isin(rare.index)]
        if out[label].nunique() < 2:
            raise DataError(
                f"fewer than 2 classes have at least {min_class_rows} rows; collect more data"
            )
    if len(out) < MIN_ROWS:
        raise DataError(f"only {len(out)} usable rows; need at least {MIN_ROWS}")
    return out, report


def _strata(label: pd.Series, task: Task) -> pd.Series | None:
    if task.is_classification:
        return label
    if len(label) < REGRESSION_STRATA * MIN_ROWS_PER_STRATUM:
        return None
    # Quantile bins keep the target's distribution the same in both splits.
    return pd.qcut(label.rank(method="first"), REGRESSION_STRATA, labels=False)


def split(
    frame: pd.DataFrame,
    label: str,
    task: Task,
    test_size: float,
    seed: int,
    groups: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split (by class, or by target quantile for regression).

    ``groups`` (for example, quasi-identifier combinations) replaces the default strata, so
    every group is divided between the splits in proportion instead of by chance.
    """
    if groups is not None:
        train, test = train_test_split(
            frame, test_size=test_size, random_state=seed, stratify=groups
        )
        return train, test
    size: float | int = test_size
    if task.is_classification:
        # Every class needs a row on each side; small data with many classes needs a larger test.
        classes = frame[label].nunique()
        size = max(int(np.ceil(test_size * len(frame))), classes)
    train, test = train_test_split(
        frame, test_size=size, random_state=seed, stratify=_strata(frame[label], task)
    )
    return train, test
