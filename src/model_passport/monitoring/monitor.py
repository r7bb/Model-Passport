"""Check a live batch against a passport and append the result as a signed event."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from model_passport.auditors.artifact import UnsafeArtifactError, safe_load_pickle
from model_passport.core import identity
from model_passport.core.events import append_event
from model_passport.core.schema import ArtifactKind, DatasetInfo, Passport, SplitRole
from model_passport.monitoring.drift import (
    DEFAULT_ALPHA,
    DEFAULT_PSI_THRESHOLD,
    DriftReport,
    check_drift,
)
from model_passport.scanners.base import ScanError, load_table, string_columns

DRIFT_EVENT = "drift_check"
DEFAULT_DEGRADATION_TOLERANCE = 0.05


class MonitorError(Exception):
    """Raised when a batch cannot be checked against the passport."""


@dataclass
class MonitorResult:
    event: dict[str, Any]
    drift: DriftReport
    performance: dict[str, Any] | None

    @property
    def retrain_recommended(self) -> bool:
        return bool(self.event["payload"]["retrain_recommended"])


def _artifact_path(passport: Passport, sha256: str, kind: ArtifactKind) -> str:
    for artifact in passport.artifacts:
        if artifact.sha256 == sha256 and artifact.kind is kind:
            return artifact.path
    raise MonitorError(f"no {kind.value} artifact with hash {sha256[:12]}")


def _verified_path(root: Path, passport: Passport, sha256: str, kind: ArtifactKind) -> Path:
    path = root / _artifact_path(passport, sha256, kind)
    if not path.is_file():
        raise MonitorError(f"{path} is missing")
    if identity.sha256_file(path) != sha256:
        raise MonitorError(f"{path} changed since the passport was built; rebuild first")
    return path


def _reference_dataset(passport: Passport, name: str | None) -> DatasetInfo:
    """The named dataset, or the training split when no name is given."""
    for dataset in passport.datasets:
        if dataset.name == name if name else dataset.split_role is SplitRole.TRAIN:
            return dataset
    raise MonitorError(f"no reference dataset {name or '(train split)'} in passport")


def _performance(
    root: Path, passport: Passport, batch: Any, label: str, tolerance: float
) -> dict[str, Any] | None:
    """Accuracy on a labeled batch versus the passport's test accuracy."""
    model_info = passport.model
    if model_info is None or label not in batch.columns:
        return None
    path = _verified_path(root, passport, model_info.artifact_sha256, ArtifactKind.MODEL)
    try:
        model = safe_load_pickle(path)
    except UnsafeArtifactError as exc:
        return {"error": str(exc)}
    labeled = batch.dropna(subset=[label])
    try:
        predictions = np.asarray(model.predict(labeled.drop(columns=[label]))).astype(str)
    except Exception as exc:  # noqa: BLE001 - a bad batch must not crash monitoring
        return {"error": f"prediction failed: {exc}"}
    accuracy = float(np.mean(predictions == labeled[label].astype(str).to_numpy()))
    baseline = passport.metrics.get("test", {}).get("accuracy")
    degraded = baseline is not None and accuracy < baseline - tolerance
    return {
        "label": label,
        "rows": len(labeled),
        "accuracy": round(accuracy, 4),
        "baseline_accuracy": baseline,
        "tolerance": tolerance,
        "degraded": degraded,
    }


def monitor_batch(
    passport_path: Path,
    batch_path: Path,
    root: Path,
    private_key: identity.Ed25519PrivateKey,
    reference: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    degradation_tolerance: float = DEFAULT_DEGRADATION_TOLERANCE,
    write: bool = True,
) -> MonitorResult:
    raw = json.loads(passport_path.read_text(encoding="utf-8"))
    passport = Passport.model_validate(raw)
    if identity.public_key_fingerprint(private_key.public_key()) != (
        passport.identity.public_key_fingerprint
    ):
        raise MonitorError("signing key does not match the key that signed this passport")

    dataset = _reference_dataset(passport, reference)
    reference_path = _verified_path(root, passport, dataset.sha256, ArtifactKind.DATASET)

    model = passport.model
    input_schema = (model.input_schema if model else None) or {}
    labels = list((model.output_schema if model else None) or {})
    text_cols = string_columns(input_schema)
    try:
        reference_frame = load_table(reference_path, text_cols)
        batch = load_table(batch_path, text_cols)
    except ScanError as exc:
        raise MonitorError(str(exc)) from exc

    features = list(input_schema) or None
    drift = check_drift(
        reference_frame, batch, features, exclude=labels, alpha=alpha, psi_threshold=psi_threshold
    )
    performance: dict[str, Any] | None = None
    if labels and not drift.schema.ok:
        performance = {"skipped": "batch does not match the model's input schema"}
    elif labels:
        performance = _performance(root, passport, batch, labels[0], degradation_tolerance)
    degraded = bool(performance and performance.get("degraded"))
    payload = {
        **drift.to_payload(),
        "batch": batch_path.name,
        "batch_sha256": identity.sha256_file(batch_path),
        "reference": dataset.name,
        "reference_sha256": dataset.sha256,
        "performance": performance,
        "retrain_recommended": drift.drift_detected or degraded or not drift.schema.ok,
    }
    event = append_event(raw, DRIFT_EVENT, payload, private_key)
    if write:
        passport_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return MonitorResult(event=event, drift=drift, performance=performance)
