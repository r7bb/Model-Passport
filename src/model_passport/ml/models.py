"""Candidate model families and cross-validated selection for any tabular task.

Three families are tried, simplest first: a regularized linear model, gradient-boosted trees,
and a random forest. Each is tuned by grid search with stratified k-fold cross-validation, and
every preprocessing step (cleaning, imputation, scaling, encoding) sits inside the pipeline, so
each fold is fitted on its own training rows only.

The winner is picked in two steps:

1. Candidates whose cross-validated train-minus-validation gap exceeds ``max_gap`` are set
   aside: a model that memorizes its training rows leaks membership and fails the passport's
   privacy gate however accurate it looks.
2. Among the rest, the simplest family whose score is within one standard error of the best
   wins (the one-standard-error rule), because differences smaller than the CV noise are not
   evidence that the more complex model is better.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from model_passport.ml.features import FeaturePlan, TableCleaner, infer_plan
from model_passport.ml.task import DataError, Task, infer_task

MISSING = "__missing__"
MIN_CATEGORY_ROWS = 5  # rarer categories are pooled as "infrequent"
LINEAR_MAX_CATEGORIES = 50
TREE_MAX_CATEGORIES = 254  # gradient boosting supports at most 255 bins, one kept for missing
MIN_ROWS_FOR_TREES = 100
SEARCH_MAX_ROWS = 20_000


class Family(StrEnum):
    """Model families, in order of increasing complexity."""

    LINEAR = "linear"
    BOOSTING = "boosting"
    FOREST = "forest"


class Budget(StrEnum):
    FULL = "full"
    QUICK = "quick"


_GRIDS: dict[tuple[Family, Budget], dict[str, list[Any]]] = {
    (Family.BOOSTING, Budget.FULL): {
        "model__learning_rate": [0.05, 0.1],
        "model__max_leaf_nodes": [7, 15, 31],
        "model__min_samples_leaf": [20, 50],
        "model__l2_regularization": [0.0, 1.0],
    },
    (Family.BOOSTING, Budget.QUICK): {
        "model__learning_rate": [0.1],
        "model__max_leaf_nodes": [7, 15],
        "model__min_samples_leaf": [20],
    },
    (Family.FOREST, Budget.FULL): {
        "model__min_samples_leaf": [3, 10, 30],
        "model__max_features": ["sqrt", 0.5],
    },
    (Family.FOREST, Budget.QUICK): {"model__min_samples_leaf": [5, 20]},
}
_LINEAR_STRENGTH = {
    Budget.FULL: [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    Budget.QUICK: [0.01, 0.1, 1.0, 10.0],
}


def param_grid(family: Family, task: Task, budget: Budget) -> dict[str, list[Any]]:
    if family is Family.LINEAR:
        strengths = _LINEAR_STRENGTH[budget]
        if task.is_classification:
            return {"model__C": strengths}
        # Ridge's alpha is the inverse of C: larger means more regularization.
        return {"model__alpha": sorted(1.0 / c for c in strengths)}
    return _GRIDS[(family, budget)]


def _categorical_steps(encoder: Any) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING)),
            ("encode", encoder),
        ]
    )


def _linear_features(plan: FeaturePlan) -> ColumnTransformer:
    numeric = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
        ]
    )
    encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=MIN_CATEGORY_ROWS,
        max_categories=LINEAR_MAX_CATEGORIES,
    )
    return ColumnTransformer(
        [
            ("num", numeric, plan.numeric_outputs),
            ("cat", _categorical_steps(encoder), plan.categorical_outputs),
        ],
        sparse_threshold=0.0,
    )


def _tree_features(plan: FeaturePlan, impute_numeric: bool) -> ColumnTransformer:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,  # gradient boosting treats negative categories as missing
        min_frequency=MIN_CATEGORY_ROWS,
        max_categories=TREE_MAX_CATEGORIES,
    )
    numeric: Any = (
        SimpleImputer(strategy="median", keep_empty_features=True)
        if impute_numeric
        else "passthrough"
    )
    return ColumnTransformer(
        [
            ("num", numeric, plan.numeric_outputs),
            ("cat", _categorical_steps(encoder), plan.categorical_outputs),
        ]
    )


def _estimator(family: Family, task: Task, plan: FeaturePlan, seed: int, budget: Budget) -> Any:
    classify = task.is_classification
    if family is Family.LINEAR:
        return LogisticRegression(max_iter=5000) if classify else Ridge()
    if family is Family.BOOSTING:
        mask = [False] * len(plan.numeric_outputs) + [True] * len(plan.categorical_outputs)
        booster = HistGradientBoostingClassifier if classify else HistGradientBoostingRegressor
        return booster(
            max_iter=500 if budget is Budget.FULL else 200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            categorical_features=mask if any(mask) else None,
            random_state=seed,
        )
    forest = RandomForestClassifier if classify else RandomForestRegressor
    return forest(n_estimators=300 if budget is Budget.FULL else 100, random_state=seed, n_jobs=1)


def build_pipeline(
    family: Family, task: Task, plan: FeaturePlan, seed: int = 0, budget: Budget = Budget.FULL
) -> Pipeline:
    """Cleaning, feature encoding, and an estimator for ``family``, as one scikit-learn pipeline."""
    features = (
        _linear_features(plan)
        if family is Family.LINEAR
        else _tree_features(plan, impute_numeric=family is Family.FOREST)
    )
    return Pipeline(
        [
            ("clean", TableCleaner(plan)),
            ("features", features),
            ("model", _estimator(family, task, plan, seed, budget)),
        ]
    )


def overfit_pipeline(task: Task, plan: FeaturePlan, seed: int = 0) -> Pipeline:
    """A deliberately memorizing model (fully grown trees, no bootstrap) for the unsafe demo."""
    forest = RandomForestClassifier if task.is_classification else RandomForestRegressor
    return Pipeline(
        [
            ("clean", TableCleaner(plan)),
            ("features", _tree_features(plan, impute_numeric=True)),
            ("model", forest(n_estimators=200, bootstrap=False, random_state=seed, n_jobs=-1)),
        ]
    )


# --- Selection ------------------------------------------------------------------------------


def scoring_for(task: Task) -> tuple[str, str]:
    """(scorer used to rank candidates, scorer used for the train-minus-validation gap).

    Ranking uses a proper scoring rule (log loss, or RMSE for regression): it rewards
    calibrated probabilities and is far less noisy across folds than accuracy.
    """
    if task.is_classification:
        return "neg_log_loss", "accuracy"
    return "neg_root_mean_squared_error", "r2"


@dataclass(frozen=True)
class Trial:
    """Cross-validated results of one family and hyperparameter setting."""

    family: Family
    params: dict[str, Any]
    score: float  # higher is better (scikit-learn's negated losses)
    score_std: float
    fit_train: float
    fit_validation: float
    folds: int

    @property
    def gap(self) -> float:
        return self.fit_train - self.fit_validation

    @property
    def score_sem(self) -> float:
        return self.score_std / math.sqrt(self.folds)

    def summary(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "params": {k.removeprefix("model__"): v for k, v in self.params.items()},
            "score": round(self.score, 5),
            "score_std": round(self.score_std, 5),
            "fit_validation": round(self.fit_validation, 4),
            "gap": round(self.gap, 4),
        }


@dataclass
class Selection:
    """The fitted winner and the evidence for choosing it."""

    model: Pipeline
    task: Task
    plan: FeaturePlan
    chosen: Trial
    reason: str
    scoring: str
    fit_metric: str
    max_gap: float
    trials: list[Trial] = field(default_factory=list)
    search_rows: int = 0

    @property
    def algorithm(self) -> str:
        return type(self.model.named_steps["model"]).__name__

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {k.removeprefix("model__"): v for k, v in self.chosen.params.items()}

    def cv_metrics(self) -> dict[str, float]:
        """The chosen model's cross-validated scores, named for the passport's metrics."""
        loss = self.scoring.removeprefix("neg_")
        return {
            f"{loss}_mean": round(-self.chosen.score, 4),
            f"{loss}_std": round(self.chosen.score_std, 4),
            f"{self.fit_metric}_mean": round(self.chosen.fit_validation, 4),
            f"{self.fit_metric}_gap": round(self.chosen.gap, 4),
            "folds": float(self.chosen.folds),
        }

    def leaderboard(self) -> list[dict[str, Any]]:
        """Best setting of each family, best first."""
        best: dict[Family, Trial] = {}
        for trial in self.trials:
            if trial.family not in best or trial.score > best[trial.family].score:
                best[trial.family] = trial
        ranked = sorted(best.values(), key=lambda t: -t.score)
        return [
            {**t.summary(), "within_gap": t.gap <= self.max_gap, "chosen": t is self.chosen}
            for t in ranked
        ]

    def describe(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "family": self.chosen.family.value,
            "algorithm": self.algorithm,
            "reason": self.reason,
            "scoring": self.scoring,
            "gap_metric": self.fit_metric,
            "max_gap": self.max_gap,
            "search_rows": self.search_rows,
            "leaderboard": self.leaderboard(),
            "features": self.plan.describe(),
        }


@dataclass(frozen=True)
class SearchSettings:
    """How hard and how carefully to search.

    ``max_gap`` is the largest cross-validated train-minus-validation gap (accuracy, or R² for
    regression) a candidate may have; keep it at or below the policy's
    ``generalization_gap_max`` warn level.
    """

    families: tuple[Family, ...] = tuple(Family)
    folds: int = 5
    max_gap: float = 0.05
    seed: int = 0
    budget: Budget = Budget.FULL
    n_jobs: int | None = -1


def cv_splitter(y: pd.Series, task: Task, folds: int, seed: int) -> StratifiedKFold | KFold:
    """k-fold CV with k reduced when the smallest class (or the data) is too small for ``folds``."""
    if task.is_classification:
        smallest = int(y.value_counts().min())
        if smallest < 2:
            raise DataError("every class needs at least 2 training rows for cross-validation")
        return StratifiedKFold(max(2, min(folds, smallest)), shuffle=True, random_state=seed)
    return KFold(max(2, min(folds, len(y) // 2)), shuffle=True, random_state=seed)


def _search_sample(
    X: pd.DataFrame, y: pd.Series, task: Task, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Tune on a stratified sample of very large data; the winner is refit on every row."""
    if len(X) <= SEARCH_MAX_ROWS:
        return X, y
    stratify = y if task.is_classification else None
    X_s, _, y_s, _ = train_test_split(
        X, y, train_size=SEARCH_MAX_ROWS, random_state=seed, stratify=stratify
    )
    return X_s, y_s


def _trials(family: Family, results: dict[str, Any], folds: int) -> list[Trial]:
    return [
        Trial(
            family=family,
            params=dict(params),
            score=float(results["mean_test_score"][i]),
            score_std=float(results["std_test_score"][i]),
            fit_train=float(results["mean_train_fit"][i]),
            fit_validation=float(results["mean_test_fit"][i]),
            folds=folds,
        )
        for i, params in enumerate(results["params"])
    ]


def choose(trials: list[Trial], max_gap: float) -> tuple[Trial, str]:
    """Gap filter, then the simplest family within one standard error of the best score."""
    finite = [t for t in trials if math.isfinite(t.score) and math.isfinite(t.gap)]
    if not finite:
        raise DataError("every candidate model failed to train; see the warnings above")
    eligible = [t for t in finite if t.gap <= max_gap]
    if eligible:
        pool, note = eligible, f"train-validation gap within {max_gap}"
    else:
        pool = [min(finite, key=lambda t: t.gap)]
        note = f"no candidate had a gap within {max_gap}; chose the least overfit"
    best = max(pool, key=lambda t: t.score)
    threshold = best.score - best.score_sem
    for family in Family:
        members = [t for t in pool if t.family is family]
        if members:
            leader = max(members, key=lambda t: t.score)
            if leader.score >= threshold:
                simpler = "" if leader is best else "; simplest family within one standard error"
                return leader, f"best cross-validated score with {note}{simpler}"
    return best, f"best cross-validated score with {note}"  # pragma: no cover - loop returns


def select_model(
    frame: pd.DataFrame,
    label: str,
    task: Task | None = None,
    settings: SearchSettings | None = None,
) -> Selection:
    """Tune every family on ``frame``, choose one, and refit it on all rows."""
    s = settings or SearchSettings()
    seed, budget = s.seed, s.budget
    y = frame[label]
    X = frame.drop(columns=[label])
    task = task or infer_task(y)
    plan = infer_plan(frame, label)
    if not plan.roles:
        raise DataError(f"no usable feature columns; dropped: {plan.dropped}")
    X_search, y_search = _search_sample(X, y, task, seed)
    cv = cv_splitter(y_search, task, s.folds, seed)
    scoring, fit_metric = scoring_for(task)
    trials: list[Trial] = []
    for family in s.families:
        if family is not Family.LINEAR and len(X_search) < MIN_ROWS_FOR_TREES:
            continue
        search = GridSearchCV(
            build_pipeline(family, task, plan, seed, budget),
            param_grid(family, task, budget),
            cv=cv,
            scoring={"score": scoring, "fit": fit_metric},
            refit=False,
            return_train_score=True,
            error_score=np.nan,
            n_jobs=s.n_jobs,
        )
        with warnings.catch_warnings():
            # Failed or unconverged settings score NaN or worse and simply lose the search.
            warnings.simplefilter("ignore", (ConvergenceWarning, FitFailedWarning, UserWarning))
            search.fit(X_search, y_search)
        trials.extend(_trials(family, search.cv_results_, cv.get_n_splits()))
    chosen, reason = choose(trials, s.max_gap)
    model = build_pipeline(chosen.family, task, plan, seed, budget).set_params(**chosen.params)
    model.fit(X, y)
    return Selection(
        model=model,
        task=task,
        plan=plan,
        chosen=chosen,
        reason=reason,
        scoring=scoring,
        fit_metric=fit_metric,
        max_gap=s.max_gap,
        trials=trials,
        search_rows=len(X_search),
    )


def fit_fixed(
    frame: pd.DataFrame,
    label: str,
    family: Family,
    task: Task | None = None,
    settings: SearchSettings | None = None,
) -> Selection:
    """Cross-validate and fit one fixed pipeline, recording the same evidence as a search.

    ``family`` is the forest for the unsafe demo's deliberately memorizing model.
    """
    s = settings or SearchSettings()
    y = frame[label]
    X = frame.drop(columns=[label])
    task = task or infer_task(y)
    plan = infer_plan(frame, label)
    model = overfit_pipeline(task, plan, s.seed)
    cv = cv_splitter(y, task, s.folds, s.seed)
    scoring, fit_metric = scoring_for(task)
    scores = cross_validate(
        model, X, y, cv=cv, scoring={"score": scoring, "fit": fit_metric}, return_train_score=True
    )
    trial = Trial(
        family=family,
        params={"model__bootstrap": False, "model__max_depth": None},
        score=float(np.mean(scores["test_score"])),
        score_std=float(np.std(scores["test_score"])),
        fit_train=float(np.mean(scores["train_fit"])),
        fit_validation=float(np.mean(scores["test_fit"])),
        folds=cv.get_n_splits(),
    )
    model.fit(X, y)
    return Selection(
        model=model,
        task=task,
        plan=plan,
        chosen=trial,
        reason="fixed by configuration (deliberately unregularized)",
        scoring=scoring,
        fit_metric=fit_metric,
        max_gap=s.max_gap,
        trials=[trial],
        search_rows=len(X),
    )
