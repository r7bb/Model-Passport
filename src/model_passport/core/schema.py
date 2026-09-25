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
    PASS = "pass"  # noqa: S105 - a verdict, not a password
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
    command: list[str] = Field(default_factory=list)
    script_path: str
    script_sha256: Sha256Hex
    git_commit: str | None = None
    git_dirty: bool | None = Field(
        default=None, description="True if the script had uncommitted changes when run."
    )
    parameters: dict[str, Any] = Field(default_factory=dict)
    inputs: list[ArtifactRef] = Field(default_factory=list)
    outputs: list[ArtifactRef] = Field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    exit_code: int | None = None


class RunInfo(_Strict):
    """The ``passport run`` invocation that produced the pipeline section."""

    run_id: UUID
    started_at: datetime
    ended_at: datetime
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    mlflow_run_id: str | None = None


class ChangeStatus(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


class DatasetChange(_Strict):
    name: str
    status: ChangeStatus
    previous_sha256: Sha256Hex | None = None
    current_sha256: Sha256Hex | None = None
    previous_rows: int | None = None
    current_rows: int | None = None


class RevisionInfo(_Strict):
    """Links a rebuilt passport (new data, retraining) to the version it supersedes."""

    previous_passport_id: UUID
    previous_version: str
    previous_merkle_root: Sha256Hex
    previous_created_at: datetime
    sequence: int = Field(default=1, ge=1, description="1 for the first rebuild, then 2, ...")
    reason: str | None = None
    added_artifacts: list[str] = Field(default_factory=list)
    removed_artifacts: list[str] = Field(default_factory=list)
    changed_artifacts: list[str] = Field(default_factory=list)
    dataset_changes: list[DatasetChange] = Field(default_factory=list)
    metric_deltas: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="split -> metric -> current minus previous"
    )


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
    details: dict[str, Any] = Field(
        default_factory=dict, description="Scores and rates only (e.g. hit_rate, confidence)."
    )


SEVERITY_ORDER = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


def at_least(severity: Severity, minimum: Severity) -> bool:
    return SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(minimum)


class RiskyCombination(_Strict):
    columns: list[str]
    unique_fraction: float


class ReidentificationResult(_Strict):
    dataset: str
    rows: int
    quasi_identifiers: list[str]
    k_anonymity: int | None = Field(description="Smallest equivalence class size.")
    unique_fraction: float
    sensitive_column: str | None = None
    l_diversity: int | None = None
    risky_combinations: list[RiskyCombination] = Field(default_factory=list)


class LeakageResult(_Strict):
    attack: str = "loss_threshold"
    mia_auc: float
    tpr_at_low_fpr: float
    low_fpr: float = 0.01
    members: int
    nonmembers: int
    generalization_gap: float | None = None
    gap_metric: str | None = None


ENTITY_AUDIT_TAXONOMY = (
    "OWASP LLM02:2025 Sensitive Information Disclosure",
    "NIST AI 100-2 E2025 privacy attacks (membership inference, data extraction)",
)


class MethodMetrics(_Strict):
    """How well one entity-level attack separates trained entities from controls."""

    method: str
    auc: float
    tpr_at_fpr: dict[str, float] = Field(
        default_factory=dict, description="True-positive rate at each false-positive rate."
    )
    entity_type: str | None = Field(default=None, description="None for all types together.")


class EntityFinding(_Strict):
    """One sensitive entity in the training data and how strongly the model memorized it.

    ``masked_value`` and ``fingerprint`` identify the entity without storing it; ``record`` and
    ``span`` locate it in the training data so remediation can act on it.
    """

    record: str
    span: tuple[int, int]
    entity_type: str
    masked_value: str
    fingerprint: str | None = Field(default=None, description="Keyed hash of the value.")
    score: float = Field(description="Primary method's membership score (higher = more likely).")
    p_value: float = Field(description="Share of same-type controls scoring at least as high.")
    q_value: float = Field(description="Benjamini-Hochberg adjusted p-value.")
    likelihood: float = Field(ge=0, le=1)
    impact: float = Field(ge=0, le=1, description="Harm if this type of value leaks.")
    risk: float = Field(ge=0, le=10, description="CVSS-style score: 10 x likelihood x impact.")
    severity: Severity
    extraction_rate: float | None = Field(
        default=None, description="Share of sampled completions that reproduced the value."
    )
    confirmed: bool | None = Field(
        default=None, description="High and critical findings are re-tested independently."
    )
    exposure: float | None = Field(
        default=None, description="Confirmation exposure in bits: log2(N + 1) - log2(rank)."
    )
    confirmation_p: float | None = None


class EntityAuditResult(_Strict):
    """Entity-level membership inference audit (EL-MIA) of a language model."""

    model: str
    access: str = Field(description="logprobs (likelihood attacks) or generation (probing).")
    primary_method: str
    methods: list[str]
    references: int
    entities_audited: int
    controls: int
    auc: float | None = Field(default=None, description="Primary method, all entity types.")
    tpr_at_fpr: dict[str, float] = Field(default_factory=dict)
    metrics: list[MethodMetrics] = Field(default_factory=list)
    severity_counts: dict[str, int] = Field(default_factory=dict)
    findings: list[EntityFinding] = Field(
        default_factory=list, description="Significant or high-risk entities, riskiest first."
    )
    fdr: float = 0.05
    taxonomy: list[str] = Field(default_factory=lambda: list(ENTITY_AUDIT_TAXONOMY))


class PrivacyReport(_Strict):
    datasets_scanned: list[str] = Field(default_factory=list)
    data_findings: list[Finding] = Field(default_factory=list)
    reidentification: list[ReidentificationResult] = Field(default_factory=list)
    leakage: LeakageResult | None = None
    entity_audit: EntityAuditResult | None = None


class DependencyAudit(StrEnum):
    OK = "ok"
    NOT_RUN = "not_run"
    ERROR = "error"


class SecurityReport(_Strict):
    files_scanned_for_secrets: int | None = None
    secret_findings: list[Finding] = Field(default_factory=list)
    artifacts_scanned: list[str] = Field(default_factory=list)
    artifact_findings: list[Finding] = Field(default_factory=list)
    dependency_audit: DependencyAudit = DependencyAudit.NOT_RUN
    dependency_audit_message: str = ""
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
    run: RunInfo | None = None
    environment: Environment | None = None
    metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="split -> metric name -> value"
    )
    privacy_report: PrivacyReport | None = None
    security_report: SecurityReport | None = None
    policy: PolicyResult | None = None
    declared: Declared = Field(default_factory=Declared)
    lineage_links: list[UUID] = Field(default_factory=list)
    revision: RevisionInfo | None = None
    events: list[LifecycleEvent] = Field(default_factory=list)
