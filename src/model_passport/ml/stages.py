"""Ready-made pipeline stages for any tabular dataset: preprocess, train, evaluate.

``passport init --data my.csv --label target`` writes three small scripts that call ``run``;
``passport run`` executes them with the parameters from ``passport.yaml``. Every stage reads
and writes plain files, so each can be replaced by your own script at any time.

- **preprocess** drops rows without a label, duplicates, and classes too rare to learn; drops
  columns the PII scan flags as direct identifiers; buckets quasi-identifiers; and makes a
  stratified train/test split. Its privacy rules go to a manifest reused for new batches.
- **train** tunes several model families with cross-validation and keeps the best one that
  does not overfit (see ``model_passport.ml.models``).
- **evaluate** reports task-appropriate metrics next to a trivial baseline.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from model_passport.auditors.artifact import safe_load_pickle
from model_passport.core.identity import sha256_file
from model_passport.ml.features import read_table
from model_passport.ml.metrics import evaluate as evaluate_model
from model_passport.ml.models import Budget, Family, SearchSettings, Selection, fit_fixed
from model_passport.ml.models import select_model as select
from model_passport.ml.privacy import (
    apply_generalization,
    detect_identifiers,
    fit_generalization,
    quasi_identifiers,
    suppress_small_groups,
)
from model_passport.ml.task import DataError, RowReport, Task, clean_rows, infer_task, split
from model_passport.runtime import params

Params = Mapping[str, Any]

PREPROCESS_DEFAULTS: dict[str, Any] = {
    "input": "data/raw.csv",
    "label": None,
    "task": "auto",
    "train_out": "data/train.csv",
    "test_out": "data/test.csv",
    "manifest_out": "data/preprocess.json",
    "drop_identifiers": True,
    "identifiers": [],  # always dropped, in addition to the ones the PII scan finds
    "generalize": True,
    "quasi_identifiers": "auto",  # or a list; "auto" suggests them from column names
    "bucket_width": 10,
    "min_k": 5,  # every group sharing quasi-identifier values has at least this many records
    "max_suppression": 0.01,  # share of records that may be removed to reach min_k
    "test_size": 0.25,
    "seed": 7,
}
TRAIN_DEFAULTS: dict[str, Any] = {
    "train": "data/train.csv",
    "manifest": "data/preprocess.json",
    "model_out": "models/model.pkl",
    "info_out": "models/model_info.json",
    "cv_out": "models/cv_metrics.json",
    "selection_out": "models/selection.json",
    "label": None,
    "model": "auto",  # auto | linear | boosting | forest | overfit (the unsafe demo)
    "cv_folds": 5,
    "max_gap": 0.05,  # keep at or below policy.yaml's generalization_gap_max warn level
    "search": "full",  # full | quick
    "seed": 7,
}
EVALUATE_DEFAULTS: dict[str, Any] = {
    "model": "models/model.pkl",
    "info": "models/model_info.json",
    "train": "data/train.csv",
    "test": "data/test.csv",
    "label": None,
    "positive": None,  # binary positive class; defaults to the second class in sorted order
    "out": "models/metrics.json",
}


def _write_json(path: str | Path, content: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(content, indent=2, default=str) + "\n", encoding="utf-8")


# --- preprocess -----------------------------------------------------------------------------


def apply_manifest(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.DataFrame:
    """Row-wise privacy transforms recorded by ``preprocess``; safe for a single new batch."""
    dropped = [c for c in manifest["dropped_identifiers"] if c in frame.columns]
    return apply_generalization(frame.drop(columns=dropped), manifest["generalization"])


def _split_target(p: Params) -> int:
    """Group size on the whole table that leaves at least ``min_k`` in the smaller split."""
    min_k: int = p["min_k"]
    smaller: float = min(p["test_size"], 1 - p["test_size"])
    return math.ceil(min_k / smaller)


def _split_anonymously(
    frame: pd.DataFrame, manifest: Mapping[str, Any], task: Task, p: Params
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Split so each quasi-identifier group reaches ``min_k`` in both files.

    Records in groups too small for that are removed first (at most ``max_suppression`` of
    them, by construction of the rules); each remaining group is then divided between train
    and test in proportion, and rounding stragglers are removed per file.
    """
    qi = manifest["quasi_identifiers"]
    if not p["generalize"] or not qi:
        train, test = split(frame, p["label"], task, p["test_size"], p["seed"])
        return train, test, {"before_split": 0, "train": 0, "test": 0}
    kept, before = suppress_small_groups(frame, qi, _split_target(p))
    if kept[p["label"]].nunique() < 2:
        raise DataError("too few records remain after anonymization; add rows or set min_k lower")
    groups = kept[qi].astype(str).agg("|".join, axis=1)
    if task.is_classification:
        # Also balance the label within each group where there are enough rows to do so.
        with_label = groups + "|" + kept[p["label"]].astype(str)
        counts = with_label.map(with_label.value_counts())
        groups = with_label.where(counts >= 2, groups)
    train, test = split(kept, p["label"], task, p["test_size"], p["seed"], groups=groups)
    train, in_train = suppress_small_groups(train, qi, p["min_k"])
    test, in_test = suppress_small_groups(test, qi, p["min_k"])
    return train, test, {"before_split": before, "train": in_train, "test": in_test}


def build_manifest(raw: pd.DataFrame, p: Params) -> dict[str, Any]:
    """Which identifier columns to drop and how to generalize quasi-identifiers."""
    label = p["label"]
    dropped: list[str] = []
    if p["drop_identifiers"]:
        found = set(detect_identifiers(raw, keep=[label])) | set(p["identifiers"])
        dropped = [str(c) for c in raw.columns if c in found and c != label]
    rules: dict[str, Any] = {}
    columns: list[str] = []
    if p["generalize"]:
        kept = raw.drop(columns=dropped)
        columns = quasi_identifiers(kept, p["quasi_identifiers"], exclude=[label])
        rules = fit_generalization(
            kept,
            columns,
            p["bucket_width"],
            _split_target(p),
            p["max_suppression"],
            label=pd.Series(kept[label]),
        )
    return {
        "label": label,
        "dropped_identifiers": dropped,
        "quasi_identifiers": columns,
        "generalization": rules,
    }


def preprocess(p: Params) -> str:
    raw = read_table(Path(p["input"]))
    if p["label"] not in raw.columns:
        raise DataError(f"label column {p['label']!r} not in {p['input']}: {list(raw.columns)}")
    task = infer_task(pd.Series(raw[p["label"]]), p["task"])
    rows, report = clean_rows(raw, p["label"], task)
    manifest = build_manifest(rows, p)
    frame = apply_manifest(rows, manifest)
    train, test, suppressed = _split_anonymously(frame, manifest, task, p)
    for part, out in ((train, p["train_out"]), (test, p["test_out"])):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(out, index=False)
    privacy = {"min_k": p["min_k"], "suppressed": suppressed}
    _write_json(
        p["manifest_out"],
        {**manifest, "task": task.value, "rows": report.describe(), "k_anonymity": privacy},
    )
    return _preprocess_summary(task, report, manifest, len(train), len(test), suppressed)


def _generalized(rules: Mapping[str, Any]) -> str:
    parts = []
    for column, rule in sorted(rules.items()):
        if rule["kind"] == "range":
            parts.append(f"{column} (ranges of {rule['width']:g})")
        elif rule["kind"] == "prefix":
            parts.append(f"{column} (first {rule['keep']} characters)")
        elif rule["kind"] == "suppress":
            parts.append(f"{column} (removed: too identifying for this much data)")
        else:
            parts.append(f"{column} ({len(rule['keep'])} categories + other)")
    return ", ".join(parts) or "none"


def _preprocess_summary(
    task: Task,
    report: RowReport,
    manifest: Mapping[str, Any],
    train_rows: int,
    test_rows: int,
    suppressed: Mapping[str, int],
) -> str:
    removed = report.rows_in - report.rows_out
    identifiers = ", ".join(manifest["dropped_identifiers"]) or "none"
    counts = f"train {train_rows}, test {test_rows}"
    lines = [
        f"{task.value}: {report.rows_in} rows ({removed} removed), {counts}",
        f"  dropped identifiers: {identifiers}",
        f"  generalized: {_generalized(manifest['generalization'])}",
    ]
    total = sum(suppressed.values())
    if total:
        share = total / max(report.rows_out, 1)
        lines.append(f"  removed {total} records ({share:.1%}) in groups too small to be anonymous")
    return "\n".join(lines)


# --- train ----------------------------------------------------------------------------------


def _fit(p: Params, frame: pd.DataFrame) -> Selection:
    manifest = Path(p["manifest"])
    task = Task(json.loads(manifest.read_text())["task"]) if manifest.is_file() else None
    settings = SearchSettings(
        folds=p["cv_folds"], max_gap=p["max_gap"], seed=p["seed"], budget=Budget(p["search"])
    )
    if p["model"] == "overfit":
        return fit_fixed(frame, p["label"], Family.FOREST, task, settings)
    if p["model"] != "auto":
        settings = replace(settings, families=(Family(p["model"]),))
    return select(frame, p["label"], task=task, settings=settings)


def model_info(selection: Selection, label: str, schema: dict[str, str]) -> dict[str, Any]:
    """Metadata ``passport build`` copies into the passport's model section."""
    classes = getattr(selection.model, "classes_", None)
    return {
        "framework": "scikit-learn",
        "algorithm": selection.algorithm,
        "task_type": selection.task.value,
        "hyperparameters": {
            "family": selection.chosen.family.value,
            **selection.hyperparameters,
            "cv_folds": selection.chosen.folds,
            "cv_scoring": selection.scoring,
            "max_gap": selection.max_gap,
        },
        "input_schema": schema,
        "output_schema": {label: sorted(map(str, classes)) if classes is not None else "number"},
    }


def _leaderboard_lines(selection: Selection) -> list[str]:
    lines = []
    for row in selection.leaderboard():
        mark = "*" if row["chosen"] else " "
        note = "" if row["within_gap"] else "  (overfits: set aside)"
        lines.append(f"  {mark} {row['family']:<9} score {row['score']:<10} gap {row['gap']}{note}")
    return lines


def train(p: Params) -> str:
    frame = read_table(Path(p["train"]))
    if p["label"] not in frame.columns:
        raise DataError(f"label column {p['label']!r} not in {p['train']}")
    selection = _fit(p, frame)
    schema = {str(c): str(t) for c, t in frame.drop(columns=[p["label"]]).dtypes.items()}
    Path(p["model_out"]).parent.mkdir(parents=True, exist_ok=True)
    with Path(p["model_out"]).open("wb") as fh:
        pickle.dump(selection.model, fh)
    _write_json(p["info_out"], model_info(selection, p["label"], schema))
    _write_json(p["cv_out"], {"cv": selection.cv_metrics()})
    _write_json(p["selection_out"], selection.describe())

    cv = selection.cv_metrics()
    metric = selection.fit_metric
    headline = (
        f"{selection.task.value}: chose {selection.chosen.family.value} "
        f"({selection.algorithm}), {selection.chosen.folds}-fold CV {metric} "
        f"{cv[f'{metric}_mean']} (train-validation gap {cv[f'{metric}_gap']})"
    )
    lines = [
        headline,
        f"  why: {selection.reason}",
        *_leaderboard_lines(selection),
    ]
    if selection.plan.dropped:
        unused = ", ".join(f"{c} ({why})" for c, why in selection.plan.dropped.items())
        lines.append(f"  unused columns: {unused}")
    return "\n".join(lines)


# --- evaluate -------------------------------------------------------------------------------


def evaluate(p: Params) -> str:
    model = safe_load_pickle(Path(p["model"]))
    task = Task(json.loads(Path(p["info"]).read_text())["task_type"])
    metrics = {
        split_name: evaluate_model(
            model, read_table(Path(p[split_name])), p["label"], task, p["positive"]
        )
        for split_name in ("train", "test")
    }
    _write_json(p["out"], metrics)
    test = metrics["test"]
    key = "accuracy" if task.is_classification else "r2"
    gap = round(metrics["train"][key] - test[key], 4)
    if task.is_classification:
        return f"test accuracy {test['accuracy']} (baseline {test['baseline_accuracy']}), gap {gap}"
    return (
        f"test R² {test['r2']}, RMSE {test['rmse']} (baseline {test['baseline_rmse']}), gap {gap}"
    )


# --- New batches ----------------------------------------------------------------------------


def certified_manifest(passport_path: Path, root: Path) -> dict[str, Any]:
    """The preprocess manifest a passport certifies, after checking it has not changed."""
    passport = json.loads(passport_path.read_text(encoding="utf-8"))
    stage = next((s for s in passport.get("pipeline", []) if s["name"] == "preprocess"), None)
    if stage is None:
        raise DataError(f"{passport_path} has no preprocess stage; run and build first")
    path = {**PREPROCESS_DEFAULTS, **stage["parameters"]}["manifest_out"]
    recorded = next((o["sha256"] for o in stage["outputs"] if o["path"] == path), None)
    full = root / path
    if recorded is None or not full.is_file():
        raise DataError(f"{path} is not certified by {passport_path}; rebuild the passport")
    if sha256_file(full) != recorded:
        raise DataError(f"{path} changed since {passport_path} was built; rebuild the passport")
    manifest: dict[str, Any] = json.loads(full.read_text(encoding="utf-8"))
    return manifest


def prepare_batch(passport_path: Path, batch: Path, out: Path, root: Path) -> int:
    """Transform a raw batch exactly like the certified model's training data."""
    prepared = apply_manifest(read_table(batch), certified_manifest(passport_path, root))
    out.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(out, index=False)
    return len(prepared)


# --- Entry point ----------------------------------------------------------------------------

STAGES: dict[str, tuple[dict[str, Any], Callable[[Params], str]]] = {
    "preprocess": (PREPROCESS_DEFAULTS, preprocess),
    "train": (TRAIN_DEFAULTS, train),
    "evaluate": (EVALUATE_DEFAULTS, evaluate),
}


def run(stage: str, defaults: Params | None = None) -> None:
    """Run one stage with parameters from ``passport run`` over ``defaults``; exit on bad data."""
    base, function = STAGES[stage]
    p = params({**base, **(defaults or {})})
    if not p.get("label"):
        raise SystemExit(f"{stage}: set `label` (the column to predict) in the stage params")
    try:
        summary = function(p)
    except DataError as exc:
        raise SystemExit(f"{stage}: {exc}") from exc
    sys.stdout.write(summary + "\n")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "")
