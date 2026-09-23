"""Train stage: an intentionally overfit or a regularized income classifier."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_passport.runtime import params

FEATURES = [
    "age",
    "gender",
    "zip",
    "education",
    "occupation",
    "marital_status",
    "hours_per_week",
    "capital_gain",
]


def make_model(kind: str, seed: int) -> tuple[object, dict]:
    if kind == "overfit":
        hyper = {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "bootstrap": False}
        return RandomForestClassifier(random_state=seed, **hyper), hyper
    if kind == "regularized":
        hyper = {"C": 0.1, "max_iter": 1000}
        return LogisticRegression(**hyper), hyper
    raise ValueError(f"unknown model kind {kind!r}")


def main() -> None:
    p = params(
        {
            "train": "data/train.csv",
            "model_out": "models/model.pkl",
            "info_out": "models/model_info.json",
            "label": "income",
            "model": "regularized",
            "seed": 7,
        }
    )
    df = pd.read_csv(p["train"], dtype={"zip": str})
    X, y = df[FEATURES], df[p["label"]]
    numeric = [c for c in FEATURES if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in FEATURES if c not in numeric]

    estimator, hyper = make_model(p["model"], p["seed"])
    model = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("num", StandardScaler(), numeric),
                    ]
                ),
            ),
            ("clf", estimator),
        ]
    )
    model.fit(X, y)

    Path(p["model_out"]).parent.mkdir(parents=True, exist_ok=True)
    with open(p["model_out"], "wb") as fh:
        pickle.dump(model, fh)
    info = {
        "framework": "scikit-learn",
        "algorithm": type(estimator).__name__,
        "task_type": "binary-classification",
        "hyperparameters": hyper,
        "input_schema": {c: str(X[c].dtype) for c in FEATURES},
        "output_schema": {p["label"]: sorted(map(str, y.unique()))},
    }
    Path(p["info_out"]).write_text(json.dumps(info, indent=2) + "\n")
    print(f"trained {info['algorithm']} on {len(df)} rows")


if __name__ == "__main__":
    main()
