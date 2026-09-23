"""Passport schema v0.1.

Every section except ``identity`` and ``artifacts`` is optional so that passports can be
built incrementally as later phases (capture, scanners, auditors, policy) come online.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

SCHEMA_VERSION = "0.1"

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
KeyFingerprint = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


def utc_now() -> datetime:
    return datetime.now(UTC)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Verdict(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SplitRole(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ArtifactKind(StrEnum):
    DATASET = "dataset"
    MODEL = "model"
    SCRIPT = "script"
    CONFIG = "config"
    OTHER = "other"


class ArtifactRef(_Strict):
    """A hashed file covered by the passport's Merkle root."""

    path: str = Field(description="POSIX path relative to the project root.")
    sha256: Sha256Hex
    size_bytes: int = Field(ge=0)
    kind: ArtifactKind = ArtifactKind.OTHER


class Identity(_Strict):
    passport_id: UUID = Field(default_factory=uuid4)
    model_name: str
    version: str
    created_at: datetime = Field(default_factory=utc_now)
    merkle_root: Sha256Hex
    public_key_fingerprint: KeyFingerprint
    signature: str | None = Field(
        default=None, description="Base64 Ed25519 signature over the canonical passport."
    )


class ModelInfo(_Strict):
    framework: str | None = None
    algorithm: str | None = None
    task_type: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    artifact_uri: str
    artifact_sha256: Sha256Hex
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


class DatasetInfo(_Strict):
    name: str
    source: str | None = None
    license: str | None = None
    row_count: int | None = Field(default=None, ge=0)
    column_schema: dict[str, str] | None = None
    sha256: Sha256Hex
    split_role: SplitRole


class PipelineStage(_Strict):
    name: str
    script_path: str
    script_sha256: Sha256Hex
    git_commit: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None


class Environment(_Strict):
    python_version: str
    os: str
    hardware: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, str] = Field(default_factory=dict)


class Finding(_Strict):
    """A single scanner or auditor result. Never holds raw sensitive values."""

    scanner: str
    category: str
    severity: Severity
    location: str | None = Field(default=None, description="Column, file, or artifact.")
    count: int = Field(default=0, ge=0)
    message: str = ""
    masked_examples: list[str] = Field(default_factory=list)


class PrivacyReport(_Strict):
    data_findings: list[Finding] = Field(default_factory=list)
    reidentification: dict[str, Any] = Field(default_factory=dict)
    leakage: dict[str, Any] = Field(default_factory=dict)


class SecurityReport(_Strict):
    artifact_findings: list[Finding] = Field(default_factory=list)
    dependency_vulnerabilities: list[Finding] = Field(default_factory=list)


class RuleResult(_Strict):
    name: str
    threshold: Any = None
    observed: Any = None
    result: Verdict
    message: str = ""


class PolicyResult(_Strict):
    policy_sha256: Sha256Hex
    rules: list[RuleResult] = Field(default_factory=list)
    verdict: Verdict


class Declared(_Strict):
    intended_use: str | None = None
    out_of_scope_uses: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    ethical_risks: list[str] = Field(default_factory=list)
    owner: str | None = None
    contact: str | None = None


class LifecycleEvent(_Strict):
    """Append-only event (drift check, retraining). Signed individually, not by the passport."""

    event_id: UUID = Field(default_factory=uuid4)
    event_type: str
    timestamp: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)
    prev_event_sha256: Sha256Hex | None = None
    signature: str | None = None


class Passport(_Strict):
    schema_version: str = SCHEMA_VERSION
    identity: Identity
    artifacts: list[ArtifactRef] = Field(min_length=1)
    model: ModelInfo | None = None
    datasets: list[DatasetInfo] = Field(default_factory=list)
    pipeline: list[PipelineStage] = Field(default_factory=list)
    environment: Environment | None = None
    metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="split -> metric name -> value"
    )
    privacy_report: PrivacyReport | None = None
    security_report: SecurityReport | None = None
    policy: PolicyResult | None = None
    declared: Declared = Field(default_factory=Declared)
    lineage_links: list[UUID] = Field(default_factory=list)
    events: list[LifecycleEvent] = Field(default_factory=list)
