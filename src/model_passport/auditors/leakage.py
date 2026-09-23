"""Model leakage audit: loss-threshold membership inference and generalization gap."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import is_regressor as _sklearn_is_regressor
from sklearn.metrics import r2_score, roc_auc_score, roc_curve

from model_passport.core.schema import LeakageResult

EPS = 1e-12


class LeakageAuditError(Exception):
    """Raised when the model or data cannot be audited."""


def is_regressor(model: Any) -> bool:
    """True for scikit-learn regressors; False for classifiers and non-scikit-learn objects."""
    try:
        return bool(_sklearn_is_regressor(model))
    except (AttributeError, TypeError):
        return getattr(model, "_estimator_type", None) == "regressor"


def _squared_error(model: Any, features: pd.DataFrame, target: pd.Series) -> np.ndarray:
    y = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    if np.isnan(y).any():
        raise LeakageAuditError("regression label has missing or non-numeric values")
    try:
        pred = np.asarray(model.predict(features), dtype=float)
    except Exception as exc:
        raise LeakageAuditError(f"predict failed: {exc}") from exc
    return np.asarray((y - pred) ** 2, dtype=float)


def per_sample_loss(model: Any, frame: pd.DataFrame, label: str) -> np.ndarray:
    """Per-row loss: cross-entropy of the true label, or squared error for regressors."""
    if label not in frame.columns:
        raise LeakageAuditError(f"label column {label!r} not found")
    features = frame.drop(columns=[label])
    if is_regressor(model):
        return _squared_error(model, features, frame[label])
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise LeakageAuditError("model must expose predict_proba and classes_ (scikit-learn API)")
    try:
        proba = np.asarray(model.predict_proba(features))
    except Exception as exc:
        raise LeakageAuditError(f"predict_proba failed: {exc}") from exc
    classes = list(model.classes_)
    index = {c: i for i, c in enumerate(classes)}
    y = frame[label].map(lambda v: index.get(v, index.get(str(v), -1))).to_numpy()
    if (y < 0).any():
        raise LeakageAuditError(f"labels not seen by the model: {set(frame[label]) - set(classes)}")
    p_true = proba[np.arange(len(y)), y]
    return np.asarray(-np.log(np.clip(p_true, EPS, 1.0)), dtype=float)


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    allowed = tpr[fpr <= max_fpr]
    return float(allowed.max()) if allowed.size else 0.0


def loss_threshold_attack(
    model: Any,
    members: pd.DataFrame,
    nonmembers: pd.DataFrame,
    label: str,
    low_fpr: float = 0.01,
    gap_metric: str = "accuracy",
) -> LeakageResult:
    """Yeom et al. loss attack: lower loss on a record suggests it was in training.

    The attack score is the negative loss; AUC 0.5 means no measurable leakage. For regressors
    the loss is squared error and an ``accuracy`` gap is measured as R² instead.
    """
    if is_regressor(model) and gap_metric == "accuracy":
        gap_metric = "r2"
    if members.empty or nonmembers.empty:
        raise LeakageAuditError("need non-empty member and non-member sets")
    member_loss = per_sample_loss(model, members, label)
    nonmember_loss = per_sample_loss(model, nonmembers, label)
    labels = np.concatenate([np.ones(len(member_loss)), np.zeros(len(nonmember_loss))])
    scores = -np.concatenate([member_loss, nonmember_loss])
    return LeakageResult(
        mia_auc=round(float(roc_auc_score(labels, scores)), 4),
        tpr_at_low_fpr=round(tpr_at_fpr(labels, scores, low_fpr), 4),
        low_fpr=low_fpr,
        members=len(member_loss),
        nonmembers=len(nonmember_loss),
        generalization_gap=round(
            generalization_gap(model, members, nonmembers, label, gap_metric), 4
        ),
        gap_metric=gap_metric,
    )


def generalization_gap(
    model: Any, train: pd.DataFrame, test: pd.DataFrame, label: str, metric: str = "accuracy"
) -> float:
    """Train metric minus test metric (for ``log_loss``: test minus train, so higher is worse)."""
    if metric == "accuracy":

        def accuracy(frame: pd.DataFrame) -> float:
            predictions = model.predict(frame.drop(columns=[label]))
            return float(np.mean(np.asarray(predictions).astype(str) == frame[label].astype(str)))

        return accuracy(train) - accuracy(test)
    if metric == "r2":

        def r2(frame: pd.DataFrame) -> float:
            target = pd.to_numeric(frame[label], errors="coerce")
            return float(r2_score(target, model.predict(frame.drop(columns=[label]))))

        return r2(train) - r2(test)
    if metric == "log_loss":
        return float(per_sample_loss(model, test, label).mean()) - float(
            per_sample_loss(model, train, label).mean()
        )
    raise LeakageAuditError(f"unsupported gap metric {metric!r}; use accuracy, log_loss, or r2")
