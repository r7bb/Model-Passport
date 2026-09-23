"""Evaluate stage: threshold and ranking metrics plus calibration on train and test splits.

Reports accuracy and balanced accuracy (the label is imbalanced), precision, recall, F1,
ROC AUC, log loss, and the Brier score (probability calibration), alongside the majority-class
baseline accuracy so a model that only predicts "<=50K" cannot look good.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model_passport.runtime import params

DEFAULTS = {
    "model": "models/model.pkl",
    "train": "data/train.csv",
    "test": "data/test.csv",
    "label": "income",
    "positive": ">50K",
    "threshold": 0.5,
    "out": "models/metrics.json",
}


def split_metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = proba >= threshold
    values = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "log_loss": log_loss(y, proba, labels=[False, True]),
        "brier": brier_score_loss(y, proba),
        "baseline_accuracy": max(y.mean(), 1 - y.mean()),
        "positive_rate": y.mean(),
    }
    return {name: round(float(value), 4) for name, value in values.items()}


def main() -> None:
    p = params(DEFAULTS)
    # The model was produced by our own train stage in this run, so loading it is safe here.
    with Path(p["model"]).open("rb") as fh:
        model = pickle.load(fh)
    positive = list(model.classes_).index(p["positive"])

    metrics = {}
    for split in ("train", "test"):
        df = pd.read_csv(p[split], dtype={"zip": str})
        y = (df[p["label"]] == p["positive"]).to_numpy()
        proba = model.predict_proba(df.drop(columns=[p["label"]]))[:, positive]
        metrics[split] = split_metrics(y, proba, p["threshold"])
    Path(p["out"]).write_text(json.dumps(metrics, indent=2) + "\n")
    test = metrics["test"]
    print(
        f"test accuracy {test['accuracy']} (baseline {test['baseline_accuracy']}), "
        f"ROC AUC {test['roc_auc']}, F1 {test['f1']}, Brier {test['brier']}"
    )


if __name__ == "__main__":
    main()
