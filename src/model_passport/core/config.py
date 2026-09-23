"""``passport.yaml`` project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field

from model_passport.core.schema import ArtifactKind, Declared, SplitRole

CONFIG_FILENAME = "passport.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectInfo(_Strict):
    name: str
    version: str = "0.1.0"


class SigningConfig(_Strict):
    private_key: Path = Path(".passport/signing_key.pem")
    public_key: Path = Path(".passport/signing_key.pub")


class ModelInput(_Strict):
    path: Path
    metadata: Path | None = Field(
        default=None,
        description="JSON written by the train stage; its keys fill fields not set here.",
    )
    framework: str | None = None
    algorithm: str | None = None
    task_type: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


class DatasetInput(_Strict):
    path: Path
    split_role: SplitRole
    name: str | None = None
    source: str | None = None
    license: str | None = None


class ArtifactInput(_Strict):
    path: Path
    kind: ArtifactKind = ArtifactKind.OTHER


class BuildInputs(_Strict):
    """Manually declared build inputs. Phase 2 stage capture will populate these."""

    model: ModelInput
    datasets: list[DatasetInput] = Field(default_factory=list)
    artifacts: list[ArtifactInput] = Field(default_factory=list)
    metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    lineage_links: list[UUID] = Field(default_factory=list)


class StageConfig(_Strict):
    """One pipeline stage, similar to a DVC stage."""

    name: str
    cmd: str | list[str]
    script: Path | None = Field(
        default=None, description="Script to hash; defaults to the first .py file in cmd."
    )
    params: dict[str, Any] = Field(default_factory=dict)
    deps: list[Path] = Field(default_factory=list)
    outs: list[Path] = Field(default_factory=list)
    metrics: Path | None = Field(
        default=None, description="JSON file (split -> metric -> value) written by the stage."
    )


class TrackingConfig(_Strict):
    mlflow_uri: str | None = None
    mlflow_experiment: str | None = None
    dvc: bool = False


class ProjectConfig(_Strict):
    project: ProjectInfo
    signing: SigningConfig = Field(default_factory=SigningConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    declared: Declared = Field(default_factory=Declared)
    stages: list[StageConfig] = Field(default_factory=list)
    build: BuildInputs


def load_config(path: Path) -> ProjectConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ProjectConfig.model_validate(data)


CONFIG_TEMPLATE = """\
# Model Passport project configuration.
# Paths are relative to the directory containing this file.

project:
  name: {name}
  version: 0.1.0

signing:
  private_key: .passport/signing_key.pem   # never commit this file
  public_key: .passport/signing_key.pub

tracking:
  mlflow_uri: null        # e.g. http://localhost:5000 (docker compose) or ./mlruns
  mlflow_experiment: null
  dvc: false              # run `dvc add` on stage outputs

stages: []
  # - name: train
  #   cmd: python train.py
  #   params: {{max_depth: 5}}      # passed to the script as JSON in $PASSPORT_PARAMS
  #   deps: [data/train.csv]
  #   outs: [models/model.pkl]
  #   metrics: null                # optional JSON file: {{split: {{metric: value}}}}

declared:
  intended_use: null
  out_of_scope_uses: []
  known_limitations: []
  ethical_risks: []
  owner: null
  contact: null

build:
  model:
    path: models/model.pkl
    framework: null
    algorithm: null
    task_type: null
    hyperparameters: {{}}
  datasets: []
    # - path: data/train.csv
    #   split_role: train          # train | validation | test
    #   source: null
    #   license: null
  artifacts: []
    # - path: train.py
    #   kind: script               # dataset | model | script | config | other
  metrics: {{}}
    # test: {{accuracy: 0.91}}
  lineage_links: []
"""
