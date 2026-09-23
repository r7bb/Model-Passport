"""Adaptive, leakage-aware training for any tabular dataset.

from model_passport.ml import select_model, evaluate
selection = select_model(train, label="income")
metrics = evaluate(selection.model, test, "income", selection.task)
"""

from model_passport.ml.features import FeaturePlan, Role, TableCleaner, infer_plan, read_table
from model_passport.ml.metrics import evaluate
from model_passport.ml.models import (
    Budget,
    Family,
    SearchSettings,
    Selection,
    fit_fixed,
    overfit_pipeline,
    select_model,
)
from model_passport.ml.task import DataError, Task, clean_rows, infer_task, split

__all__ = [
    "Budget",
    "DataError",
    "Family",
    "FeaturePlan",
    "Role",
    "SearchSettings",
    "Selection",
    "TableCleaner",
    "Task",
    "clean_rows",
    "evaluate",
    "fit_fixed",
    "infer_plan",
    "infer_task",
    "overfit_pipeline",
    "read_table",
    "select_model",
    "split",
]
