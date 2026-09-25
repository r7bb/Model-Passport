"""Provenance capture: stage runner, git state, and environment capture."""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field

from model_passport.core import identity
from model_passport.core.config import ProjectConfig, StageConfig
from model_passport.core.schema import (
    ArtifactKind,
    ArtifactRef,
    Environment,
    PipelineStage,
    utc_now,
)

RUN_RECORD_PATH = Path(".passport/run.json")
PARAMS_ENV = "PASSPORT_PARAMS"
STAGE_ENV = "PASSPORT_STAGE"


class CaptureError(Exception):
    """Raised when a stage fails or its declared inputs or outputs are missing."""


class RunRecord(BaseModel):
    """Result of ``passport run``, consumed by ``passport build``."""

    model_config = ConfigDict(extra="forbid")

    run_id: UUID = Field(default_factory=uuid4)
    started_at: datetime
    ended_at: datetime
    stages_sha256: str = Field(description="Hash of the stages config, before overrides.")
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    stages: list[PipelineStage]
    environment: Environment
    metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    mlflow_run_id: str | None = None


# --- Git -----------------------------------------------------------------------------------


def _git(root: Path, *args: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(
            [git, *args], cwd=root, capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_commit(root: Path) -> str | None:
    return _git(root, "rev-parse", "HEAD") or None


def git_is_dirty(root: Path, path: Path) -> bool | None:
    """True if ``path`` has uncommitted changes (or is untracked); None outside a repo."""
    if git_commit(root) is None:
        return None
    status = _git(root, "status", "--porcelain", "--", str(path))
    return None if status is None else bool(status)


# --- Environment ---------------------------------------------------------------------------


def _memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def capture_environment() -> Environment:
    dependencies = {
        dist.metadata["Name"]: dist.version
        for dist in metadata.distributions()
        if dist.metadata["Name"]
    }
    return Environment(
        python_version=platform.python_version(),
        os=platform.platform(),
        hardware={
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "cpu_count": os.cpu_count(),
            "memory_bytes": _memory_bytes(),
        },
        dependencies=dict(sorted(dependencies.items(), key=lambda kv: kv[0].lower())),
    )


# --- Stage runner --------------------------------------------------------------------------


def parse_overrides(assignments: list[str]) -> dict[str, dict[str, Any]]:
    """Parse ``stage.key=value`` strings; values are YAML-typed (``3`` -> int, ``true`` -> bool)."""
    overrides: dict[str, dict[str, Any]] = {}
    for item in assignments:
        target, sep, raw = item.partition("=")
        stage, dot, key = target.partition(".")
        if not sep or not dot or not stage or not key:
            raise CaptureError(f"invalid override {item!r}; expected stage.key=value")
        overrides.setdefault(stage, {})[key] = yaml.safe_load(raw)
    return overrides


def stage_command(stage: StageConfig) -> list[str]:
    cmd = shlex.split(stage.cmd) if isinstance(stage.cmd, str) else list(stage.cmd)
    if not cmd:
        raise CaptureError(f"stage {stage.name!r} has an empty cmd")
    if cmd[0] in {"python", "python3"}:
        cmd[0] = sys.executable  # run with the interpreter that has model_passport installed
    return cmd


def stage_script(stage: StageConfig) -> Path:
    if stage.script is not None:
        return stage.script
    for token in stage_command(stage)[1:]:
        if token.endswith(".py"):
            return Path(token)
    raise CaptureError(f"stage {stage.name!r}: cannot infer script; set `script:` explicitly")


def hash_path(root: Path, path: Path, kind: ArtifactKind = ArtifactKind.OTHER) -> ArtifactRef:
    full = (root / path).resolve()
    if not full.exists():
        raise CaptureError(f"file not found: {path}")
    try:
        rel = full.relative_to(root).as_posix()
    except ValueError:
        rel = full.as_posix()
    return ArtifactRef(
        path=rel, sha256=identity.sha256_path(full), size_bytes=identity.path_size(full), kind=kind
    )


def stages_fingerprint(stages: list[StageConfig]) -> str:
    return identity.sha256_bytes(
        identity.canonical_json([s.model_dump(mode="json") for s in stages])
    )


def run_stage(
    root: Path, stage: StageConfig, params: dict[str, Any], commit: str | None
) -> PipelineStage:
    script = stage_script(stage)
    script_ref = hash_path(root, script, ArtifactKind.SCRIPT)
    inputs = [hash_path(root, dep) for dep in stage.deps]
    command = stage_command(stage)

    env = {**os.environ, PARAMS_ENV: json.dumps(params), STAGE_ENV: stage.name}
    started = utc_now()
    try:
        result = subprocess.run(command, cwd=root, env=env, check=False)
    except OSError as exc:
        raise CaptureError(f"stage {stage.name!r} could not start: {exc}") from exc
    ended = utc_now()
    if result.returncode != 0:
        raise CaptureError(f"stage {stage.name!r} failed with exit code {result.returncode}")

    missing = [str(out) for out in stage.outs if not (root / out).exists()]
    if missing:
        raise CaptureError(f"stage {stage.name!r} did not produce: {', '.join(missing)}")

    return PipelineStage(
        name=stage.name,
        # Record the interpreter by name only so local paths don't leak into the passport.
        command=[Path(command[0]).name, *command[1:]],
        script_path=script_ref.path,
        script_sha256=script_ref.sha256,
        git_commit=commit,
        git_dirty=git_is_dirty(root, script),
        parameters=params,
        inputs=inputs,
        outputs=[hash_path(root, out) for out in stage.outs],
        started_at=started,
        ended_at=ended,
        exit_code=result.returncode,
    )


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {split: {k: float(v) for k, v in values.items()} for split, values in data.items()}


StageHook = Callable[[StageConfig, PipelineStage, dict[str, dict[str, float]]], None]


def run_pipeline(
    root: Path,
    config: ProjectConfig,
    overrides: dict[str, dict[str, Any]] | None = None,
    on_stage: StageHook | None = None,
) -> RunRecord:
    """Run every configured stage in order and return the provenance record."""
    if not config.stages:
        raise CaptureError("no stages declared in passport.yaml")
    overrides = overrides or {}
    unknown = sorted(set(overrides) - {s.name for s in config.stages})
    if unknown:
        raise CaptureError(f"unknown stage(s) in overrides: {', '.join(unknown)}")

    started = utc_now()
    commit = git_commit(root)
    stages: list[PipelineStage] = []
    metrics: dict[str, dict[str, float]] = {}
    for stage in config.stages:
        params = {**stage.params, **overrides.get(stage.name, {})}
        record = run_stage(root, stage, params, commit)
        stage_metrics: dict[str, dict[str, float]] = {}
        if stage.metrics is not None:
            metrics_path = root / stage.metrics
            if not metrics_path.is_file():
                raise CaptureError(f"stage {stage.name!r} did not write metrics {stage.metrics}")
            stage_metrics = read_metrics(metrics_path)
            for split, values in stage_metrics.items():
                metrics.setdefault(split, {}).update(values)
        stages.append(record)
        if on_stage is not None:
            on_stage(stage, record, stage_metrics)

    return RunRecord(
        started_at=started,
        ended_at=utc_now(),
        stages_sha256=stages_fingerprint(config.stages),
        overrides=overrides,
        stages=stages,
        environment=capture_environment(),
        metrics=metrics,
    )


def save_run_record(root: Path, record: RunRecord) -> Path:
    path = root / RUN_RECORD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_run_record(root: Path) -> RunRecord | None:
    path = root / RUN_RECORD_PATH
    if not path.is_file():
        return None
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
