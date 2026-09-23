"""Run the configured scanners and auditors during ``passport build``."""

from __future__ import annotations

from pathlib import Path

from model_passport.core.config import PrivacyConfig
from model_passport.core.schema import ArtifactKind, ArtifactRef, DatasetInfo, PrivacyReport
from model_passport.scanners.base import ScanError, ScanTarget
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
