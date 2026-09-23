"""Run the configured scanners and auditors during ``passport build``."""

from __future__ import annotations

from pathlib import Path

from model_passport.auditors.artifact import (
    UnsafeArtifactError,
    audit_dependencies,
    is_pickle_like,
    safe_load_pickle,
    scan_model_file,
)
from model_passport.auditors.leakage import LeakageAuditError, loss_threshold_attack
from model_passport.core.config import AuditConfig, PrivacyConfig
from model_passport.core.schema import (
    ArtifactKind,
    ArtifactRef,
    DatasetInfo,
    Environment,
    Finding,
    LeakageResult,
    PrivacyReport,
    SecurityReport,
    Severity,
    SplitRole,
)
from model_passport.scanners.base import ScanError, ScanTarget, load_table, string_columns
from model_passport.scanners.data_pii import PiiScanner
from model_passport.scanners.reid_risk import ReidRiskScanner
from model_passport.scanners.secrets import SecretsScanner


class AssessmentError(Exception):
    """Raised when a configured check cannot run."""


def assess_privacy(
    root: Path, config: PrivacyConfig, datasets: list[tuple[str, DatasetInfo]]
) -> PrivacyReport:
    """PII and reidentification scans. Also fills row_count and column_schema in place."""
    report = PrivacyReport()
    if not config.enabled:
        return report
    pii = PiiScanner(sample_size=config.sample_size, use_presidio=config.presidio)
    reid = ReidRiskScanner(config.quasi_identifiers, config.sensitive_column)
    for path, info in datasets:
        target = ScanTarget(root / path, name=info.name)
        if not target.is_tabular:
            continue
        try:
            frame = target.frame
        except ScanError as exc:
            raise AssessmentError(str(exc)) from exc
        info.row_count = len(frame)
        info.column_schema = {str(c): str(t) for c, t in frame.dtypes.items()}
        report.data_findings.extend(pii.scan(target))
        report.data_findings.extend(reid.scan(target))
        report.datasets_scanned.append(info.name)
    report.reidentification = reid.results
    return report


def _dataset_path(
    explicit: Path | None, datasets: list[tuple[str, DatasetInfo]], role: SplitRole
) -> str | None:
    if explicit is not None:
        return str(explicit)
    return next((path for path, info in datasets if info.split_role is role), None)


def assess_models(
    root: Path,
    config: AuditConfig,
    model_path: str,
    artifacts: list[ArtifactRef],
    datasets: list[tuple[str, DatasetInfo]],
    environment: Environment | None,
    report: SecurityReport,
    input_schema: dict[str, str] | None = None,
) -> LeakageResult | None:
    """Artifact scan, dependency audit (into ``report``), and leakage audit (returned)."""
    if config.scan_artifacts:
        for artifact in artifacts:
            if artifact.kind is ArtifactKind.MODEL or is_pickle_like(Path(artifact.path)):
                report.artifact_findings.extend(
                    scan_model_file(root / artifact.path, label=artifact.path)
                )
                report.artifacts_scanned.append(artifact.path)

    if config.dependency_audit and environment is not None:
        status, message, vulns = audit_dependencies(environment.dependencies)
        report.dependency_audit = status
        report.dependency_audit_message = message
        report.dependency_vulnerabilities = vulns

    if not config.label_column:
        return None
    members = _dataset_path(config.members, datasets, SplitRole.TRAIN)
    nonmembers = _dataset_path(config.nonmembers, datasets, SplitRole.TEST)
    if members is None or nonmembers is None:
        raise AssessmentError("leakage audit needs member (train) and non-member (test) data")
    try:
        model = safe_load_pickle(root / model_path)
    except UnsafeArtifactError as exc:
        report.artifact_findings.append(
            Finding(
                scanner="leakage", category="AUDIT_SKIPPED", severity=Severity.HIGH,
                location=model_path, message=str(exc),
            )
        )  # fmt: skip
        return None
    text_cols = string_columns(input_schema)
    try:
        return loss_threshold_attack(
            model,
            load_table(root / members, text_cols),
            load_table(root / nonmembers, text_cols),
            config.label_column,
            gap_metric=config.gap_metric,
        )
    except (LeakageAuditError, ScanError) as exc:
        raise AssessmentError(f"leakage audit failed: {exc}") from exc


def scan_secrets(root: Path, artifacts: list[ArtifactRef]) -> tuple[int, list]:
    """Secrets scan over every non-model artifact. Returns (files scanned, findings)."""
    scanner = SecretsScanner()
    findings = []
    scanned = 0
    for artifact in artifacts:
        if artifact.kind is ArtifactKind.MODEL:
            continue
        scanned += 1
        findings.extend(scanner.scan(ScanTarget(root / artifact.path, name=artifact.path)))
    return scanned, findings
