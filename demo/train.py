"""Train stage: an intentionally overfit or a cross-validated regularized income classifier.

All preprocessing (imputation, scaling, encoding) lives inside the scikit-learn pipeline, so
cross-validation refits it on each training fold and never leaks statistics from the
held-out fold. The regularization strength is chosen by stratified k-fold grid search.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
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
C_GRID = [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
CV_SCORING = "roc_auc"
DEFAULTS = {
    "train": "data/train.csv",
    "model_out": "models/model.pkl",
    "info_out": "models/model_info.json",
    "cv_out": "models/cv_metrics.json",
    "label": "income",
    "model": "regularized",
    "cv_folds": 5,
    "seed": 7,
}


def feature_pipeline(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in FEATURES if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in FEATURES if c not in numeric]
    return ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
        ]
    )


def fit(kind: str, X: pd.DataFrame, y: pd.Series, folds: int, seed: int) -> dict[str, Any]:
    """Fit the requested model; returns the fitted pipeline, hyperparameters, and CV scores."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    if kind == "overfit":
        # Deliberately unregularized: fully grown trees, no bootstrap. Used to show leakage.
        hyper: dict[str, Any] = {
            "n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "bootstrap": False,
        }  # fmt: skip
        model = Pipeline(
            [("features", feature_pipeline(X)),
             ("clf", RandomForestClassifier(random_state=seed, n_jobs=-1, **hyper))]
        )  # fmt: skip
        scores = cross_val_score(model, X, y, cv=cv, scoring=CV_SCORING)
        model.fit(X, y)
        return {"model": model, "hyper": hyper, "cv_mean": scores.mean(), "cv_std": scores.std()}
    if kind == "regularized":
        base = Pipeline(
            [("features", feature_pipeline(X)), ("clf", LogisticRegression(max_iter=2000))]
        )
        search = GridSearchCV(base, {"clf__C": C_GRID}, cv=cv, scoring=CV_SCORING, refit=True)
        search.fit(X, y)
        best = search.best_index_
        chosen = search.best_params_["clf__C"]
        if chosen in (C_GRID[0], C_GRID[-1]):
            print(f"warning: best C={chosen} is at the edge of the search grid; widen C_GRID")
        return {
            "model": search.best_estimator_,
            "hyper": {"C": chosen, "max_iter": 2000, "C_grid": C_GRID},
            "cv_mean": search.cv_results_["mean_test_score"][best],
            "cv_std": search.cv_results_["std_test_score"][best],
        }
    raise ValueError(f"unknown model kind {kind!r}")


def main() -> None:
    p = params(DEFAULTS)
    df = pd.read_csv(p["train"], dtype={"zip": str})
    X, y = df[FEATURES], df[p["label"]]
    result = fit(p["model"], X, y, p["cv_folds"], p["seed"])
    if not np.isfinite(result["cv_mean"]):
        raise SystemExit("cross-validation produced no score")
    model = result["model"]

    Path(p["model_out"]).parent.mkdir(parents=True, exist_ok=True)
    with open(p["model_out"], "wb") as fh:
        pickle.dump(model, fh)
    info = {
        "framework": "scikit-learn",
        "algorithm": type(model.named_steps["clf"]).__name__,
        "task_type": "binary-classification",
        "hyperparameters": {**result["hyper"], "cv_folds": p["cv_folds"], "cv_scoring": CV_SCORING},
        "input_schema": {c: str(X[c].dtype) for c in FEATURES},
        "output_schema": {p["label"]: sorted(map(str, y.unique()))},
    }
    Path(p["info_out"]).write_text(json.dumps(info, indent=2) + "\n")
    cv_metrics = {
        "cv": {
            f"{CV_SCORING}_mean": round(float(result["cv_mean"]), 4),
            f"{CV_SCORING}_std": round(float(result["cv_std"]), 4),
        }
    }
    Path(p["cv_out"]).write_text(json.dumps(cv_metrics, indent=2) + "\n")
    print(
        f"trained {info['algorithm']} on {len(df)} rows; "
        f"{p['cv_folds']}-fold CV {CV_SCORING} {result['cv_mean']:.4f} ± {result['cv_std']:.4f}"
        + (f"; chose C={result['hyper']['C']}" if "C" in result["hyper"] else "")
    )


if __name__ == "__main__":
    main()
