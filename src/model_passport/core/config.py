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


class PrivacyConfig(_Strict):
    """Data scans run by ``passport build`` over every declared dataset."""

    enabled: bool = True
    sample_size: int | None = Field(
        default=10_000, description="Rows sampled for PII scanning; null scans everything."
    )
    quasi_identifiers: list[str] | None = Field(
        default=None, description="Null suggests them from column names."
    )
    sensitive_column: str | None = None
    presidio: bool = False
    scan_secrets: bool = True
    entity_audit: Path | None = Field(
        default=None,
        description="Entity-level audit JSON from `passport llm audit`; signed with the passport.",
    )


class AuditConfig(_Strict):
    """Model and artifact audits run by ``passport build``."""

    label_column: str | None = Field(
        default=None, description="Enables the leakage audit when set."
    )
    members: Path | None = Field(default=None, description="Defaults to the train dataset.")
    nonmembers: Path | None = Field(default=None, description="Defaults to the test dataset.")
    gap_metric: str = "accuracy"
    scan_artifacts: bool = True
    dependency_audit: bool = True


class ProjectConfig(_Strict):
    project: ProjectInfo
    signing: SigningConfig = Field(default_factory=SigningConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    declared: Declared = Field(default_factory=Declared)
    stages: list[StageConfig] = Field(default_factory=list)
    build: BuildInputs
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    policy: Path | None = Field(default=None, description="Policy file; null skips the gate.")


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


DATA_CONFIG_TEMPLATE = """\
# Model Passport project for {data}, predicting `{label}`.
# Created by `passport init --data`. Next: `passport run && passport build`.
# Paths are relative to the directory containing this file.

project:
  name: {name}
  version: 0.1.0

signing:
  private_key: .passport/signing_key.pem   # never commit this file
  public_key: .passport/signing_key.pub

stages:
  # Ready-made stages that adapt to the data (see model_passport.ml.stages).
  # Replace any script with your own; keep the same inputs and outputs.
  - name: preprocess
    cmd: python stages/preprocess.py
    params:
      input: {data}
      label: {label}
      drop_identifiers: true      # drop columns the PII scan flags (names, emails, IDs...)
      generalize: true            # bucket quasi-identifiers such as age and ZIP code
      quasi_identifiers: auto     # or a list, e.g. [age, zip, gender]
      test_size: 0.25
      seed: 7
    deps: [{data}]
    outs: [data/train.csv, data/test.csv, data/preprocess.json]

  - name: train
    cmd: python stages/train.py
    params:
      label: {label}
      model: auto                 # auto | linear | boosting | forest
      cv_folds: 5
      max_gap: 0.05               # reject candidates that overfit more than this
      search: full                # full | quick
      seed: 7
    deps: [data/train.csv, data/preprocess.json]
    outs: [models/model.pkl, models/model_info.json, models/cv_metrics.json, models/selection.json]
    metrics: models/cv_metrics.json

  - name: evaluate
    cmd: python stages/evaluate.py
    params:
      label: {label}
    deps: [models/model.pkl, models/model_info.json, data/train.csv, data/test.csv]
    outs: [models/metrics.json]
    metrics: models/metrics.json

declared:                         # shown in the report; fill these in
  intended_use: null
  out_of_scope_uses: []
  known_limitations: []
  ethical_risks: []
  owner: null
  contact: null

build:
  model:
    path: models/model.pkl
    metadata: models/model_info.json
  datasets:
    - path: data/train.csv
      split_role: train
      source: {data}
    - path: data/test.csv
      split_role: test
      source: {data}

privacy:
  quasi_identifiers: {quasi}
  sensitive_column: null

audit:
  label_column: {label}

policy: policy.yaml
"""

STAGE_SCRIPT_TEMPLATE = '''\
"""{stage} stage created by `passport init --data`. See model_passport.ml.stages.{stage}."""

from model_passport.ml.stages import run

if __name__ == "__main__":
    run("{stage}")
'''

POLICY_TEMPLATE = """\
# Model Passport policy gate. `passport build` exits 1 when the overall verdict is fail.
#
# Threshold forms:
#   rule: 5                        fail when violated
#   rule: {warn: 0.55, fail: 0.60} warn and fail levels
#   unsafe_pickle: warn            verdict applied when the condition is found

rules:
  pii_columns_max: 0              # columns containing direct identifiers
  min_k_anonymity: 5              # smallest group sharing the same quasi-identifier values
  unique_record_fraction_max: 0.05
  mia_auc_max: {warn: 0.55, fail: 0.60}
  generalization_gap_max: {warn: 0.05, fail: 0.10}
  secrets_found_max: 0
  unsafe_pickle: fail             # pickles importing dangerous callables (os.system, eval, ...)
  critical_cves_max: 0

# Verdict for a rule whose evidence was never collected (e.g. the leakage audit did not run).
missing_evidence: warn
"""
