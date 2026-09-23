"""Task-appropriate evaluation metrics, each reported next to a trivial baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from model_passport.ml.features import category_text
from model_passport.ml.task import Task


def _rounded(values: dict[str, Any]) -> dict[str, float]:
    return {name: round(float(value), 4) for name, value in values.items()}


def _binary(y: np.ndarray, proba: np.ndarray, classes: list[str], positive: str) -> dict[str, Any]:
    column = classes.index(positive)
    is_positive = y == positive
    pred = proba.argmax(axis=1) == column
    return {
        "precision": precision_score(is_positive, pred, zero_division=0),
        "recall": recall_score(is_positive, pred, zero_division=0),
        "f1": f1_score(is_positive, pred, zero_division=0),
        "roc_auc": roc_auc_score(is_positive, proba[:, column]),
        "brier": brier_score_loss(is_positive, proba[:, column]),
        "positive_rate": is_positive.mean(),
    }


def _classification(
    model: Any, X: pd.DataFrame, y_raw: pd.Series, positive: str | None
) -> dict[str, float]:
    # Compare labels as text, and sort classes the way scikit-learn's metrics order labels, so
    # integer classes read back from CSV (3 vs 3.0) and probability columns line up.
    names = [category_text(c) for c in model.classes_]
    order = np.argsort(names)
    classes = [names[i] for i in order]
    y = y_raw.map(category_text).to_numpy(dtype=object)
    known = np.isin(y, classes)
    X, y = X[known], y[known]
    if len(y) == 0:
        return {"unknown_label_rows": float((~known).sum())}
    proba = np.asarray(model.predict_proba(X))[:, order]
    pred = np.asarray(classes, dtype=object)[proba.argmax(axis=1)]
    shares = pd.Series(y).value_counts(normalize=True)
    values: dict[str, Any] = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "log_loss": log_loss(y, proba, labels=classes),
        "baseline_accuracy": shares.max(),  # always predicting the most common class
    }
    present = set(y)
    if len(classes) == 2 and len(present) == 2:
        values.update(_binary(y, proba, classes, positive or classes[1]))
    elif len(classes) > 2:
        values["f1_macro"] = f1_score(y, pred, average="macro", zero_division=0)
        if present == set(classes):
            values["roc_auc_ovr"] = roc_auc_score(y, proba, multi_class="ovr", labels=classes)
    if not known.all():
        values["unknown_label_rows"] = (~known).sum()
    return _rounded(values)


def _regression(model: Any, X: pd.DataFrame, y_raw: pd.Series) -> dict[str, float]:
    y = pd.to_numeric(y_raw, errors="coerce")
    usable = y.notna().to_numpy()
    y_true = y[usable].to_numpy(dtype=float)
    pred = np.asarray(model.predict(X[usable]), dtype=float)
    values: dict[str, Any] = {
        "rmse": root_mean_squared_error(y_true, pred),
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred) if len(y_true) > 1 else float("nan"),
        "baseline_rmse": float(np.std(y_true)),  # always predicting the mean
    }
    return _rounded(values)


def evaluate(
    model: Any, frame: pd.DataFrame, label: str, task: Task, positive: str | None = None
) -> dict[str, float]:
    """Metrics for ``model`` on ``frame``. Rows whose label the model never saw are counted."""
    X = frame.drop(columns=[label])
    if task.is_classification:
        return _classification(model, X, frame[label], positive)
    return _regression(model, X, frame[label])
