"""Model leakage audit: loss-threshold membership inference and generalization gap."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from model_passport.core.schema import LeakageResult

EPS = 1e-12


class LeakageAuditError(Exception):
    """Raised when the model or data cannot be audited."""


def per_sample_loss(model: Any, frame: pd.DataFrame, label: str) -> np.ndarray:
    """Cross-entropy of the true label for each row."""
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise LeakageAuditError("model must expose predict_proba and classes_ (scikit-learn API)")
    if label not in frame.columns:
        raise LeakageAuditError(f"label column {label!r} not found")
    features = frame.drop(columns=[label])
    try:
        proba = np.asarray(model.predict_proba(features))
    except Exception as exc:  # noqa: BLE001 - surface any model error as an audit error
        raise LeakageAuditError(f"predict_proba failed: {exc}") from exc
    classes = list(model.classes_)
    index = {c: i for i, c in enumerate(classes)}
    y = frame[label].map(lambda v: index.get(v, index.get(str(v), -1))).to_numpy()
    if (y < 0).any():
        raise LeakageAuditError(f"labels not seen by the model: {set(frame[label]) - set(classes)}")
    p_true = proba[np.arange(len(y)), y]
    return -np.log(np.clip(p_true, EPS, 1.0))


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

    The attack score is the negative loss; AUC 0.5 means no measurable leakage.
    """
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
    if metric == "log_loss":
        return float(per_sample_loss(model, test, label).mean()) - float(
            per_sample_loss(model, train, label).mean()
        )
    raise LeakageAuditError(f"unsupported gap metric {metric!r}; use accuracy or log_loss")
