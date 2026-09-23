"""Optional MLflow and DVC integration. Both degrade to warnings when not installed."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from model_passport.core.config import StageConfig, TrackingConfig
from model_passport.core.schema import PipelineStage


class TrackingUnavailableError(Exception):
    """Raised when a configured tracking backend cannot be used."""


def resolve_tracking_uri(uri: str, root: Path) -> str:
    """Resolve scheme-less URIs (plain paths) against the project root."""
    if urlparse(uri).scheme:
        return uri
    return (root / uri).resolve().as_uri()


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = str(value)[:6000]
    return flat


class MlflowTracker:
    """Logs a parent run for the pipeline and one nested run per stage."""

    def __init__(self, config: TrackingConfig, root: Path, project_name: str) -> None:
        if not config.mlflow_uri:
            raise TrackingUnavailableError("tracking.mlflow_uri is not set")
        try:
            from mlflow.tracking import MlflowClient  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise TrackingUnavailableError("mlflow is not installed (pip install mlflow)") from exc

        self.client = MlflowClient(tracking_uri=resolve_tracking_uri(config.mlflow_uri, root))
        name = config.mlflow_experiment or project_name
        experiment = self.client.get_experiment_by_name(name)
        self.experiment_id = (
            experiment.experiment_id if experiment else self.client.create_experiment(name)
        )
        self.parent_run_id: str | None = None

    def start(self, git_commit: str | None) -> str:
        tags = {"mlflow.runName": "passport-run", "passport.source": "model_passport"}
        if git_commit:
            tags["mlflow.source.git.commit"] = git_commit
        run = self.client.create_run(self.experiment_id, tags=tags)
        self.parent_run_id = run.info.run_id
        return self.parent_run_id

    def log_stage(
        self, stage: StageConfig, record: PipelineStage, metrics: dict[str, dict[str, float]]
    ) -> None:
        tags = {
            "mlflow.runName": stage.name,
            "mlflow.parentRunId": self.parent_run_id or "",
            "passport.script_sha256": record.script_sha256,
        }
        run = self.client.create_run(self.experiment_id, tags=tags)
        run_id = run.info.run_id
        for key, param in _flatten(record.parameters).items():
            self.client.log_param(run_id, key, param)
        for split, values in metrics.items():
            for name, metric in values.items():
                self.client.log_metric(run_id, f"{split}_{name}", metric)
                if self.parent_run_id:
                    self.client.log_metric(self.parent_run_id, f"{split}_{name}", metric)
        self.client.set_terminated(run_id)

    def finish(self, status: str = "FINISHED") -> None:
        if self.parent_run_id:
            self.client.set_terminated(self.parent_run_id, status=status)


def log_passport(
    config: TrackingConfig, root: Path, run_id: str, passport_path: Path, tags: dict[str, str]
) -> None:
    """Attach a built passport to the pipeline's MLflow run."""
    if not config.mlflow_uri:
        raise TrackingUnavailableError("tracking.mlflow_uri is not set")
    try:
        from mlflow.tracking import MlflowClient  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise TrackingUnavailableError("mlflow is not installed (pip install mlflow)") from exc
    client = MlflowClient(tracking_uri=resolve_tracking_uri(config.mlflow_uri, root))
    for key, value in tags.items():
        client.set_tag(run_id, key, value)
    client.log_artifact(run_id, str(passport_path), artifact_path="passport")


def dvc_add(root: Path, paths: list[Path]) -> None:
    """Track stage outputs with DVC (``dvc add``)."""
    if not paths:
        return
    dvc = shutil.which("dvc")
    if dvc is None:
        raise TrackingUnavailableError("dvc is not installed (pip install dvc)")
    if not (root / ".dvc").is_dir():
        raise TrackingUnavailableError("not a DVC repository (run `dvc init`)")
    result = subprocess.run(
        [dvc, "add", *[str(p) for p in paths]],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise TrackingUnavailableError(f"dvc add failed: {result.stderr.strip()}")
