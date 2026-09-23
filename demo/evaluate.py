"""Evaluate stage: accuracy, F1, and ROC AUC on the train and test splits."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from model_passport.runtime import params


def main() -> None:
    p = params(
        {
            "model": "models/model.pkl",
            "train": "data/train.csv",
            "test": "data/test.csv",
            "label": "income",
            "positive": ">50K",
            "out": "models/metrics.json",
        }
    )
    # The model was produced by our own train stage in this run, so loading it is safe here.
    with open(p["model"], "rb") as fh:
        model = pickle.load(fh)  # noqa: S301

    metrics = {}
    for split in ("train", "test"):
        df = pd.read_csv(p[split], dtype={"zip": str})
        y = df[p["label"]] == p["positive"]
        proba = model.predict_proba(df)[:, list(model.classes_).index(p["positive"])]
        pred = proba >= 0.5
        metrics[split] = {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "f1": round(float(f1_score(y, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        }
    Path(p["out"]).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
