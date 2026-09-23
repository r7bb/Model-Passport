from __future__ import annotations

from pathlib import Path

import pytest

from model_passport.core.schema import (
    ArtifactRef,
    DependencyAudit,
    Finding,
    Identity,
    LeakageResult,
    Passport,
    PrivacyReport,
    ReidentificationResult,
    SecurityReport,
    Severity,
    Verdict,
)
from model_passport.policy.engine import Policy, PolicyError, evaluate, load_policy, worst

SHA = "a" * 64


def _passport(**reports: object) -> Passport:
    return Passport(
        identity=Identity(
            model_name="m", version="1", merkle_root=SHA, public_key_fingerprint="sha256:" + SHA
        ),
        artifacts=[ArtifactRef(path="m", sha256=SHA, size_bytes=1)],
        **reports,
    )


def _pii(column: str) -> Finding:
    return Finding(
        scanner="pii", category="EMAIL", severity=Severity.HIGH, location=f"train:{column}",
        details={"direct_identifier": True},
    )  # fmt: skip


def _reid(k: int, unique: float) -> ReidentificationResult:
    return ReidentificationResult(
        dataset="train", rows=100, quasi_identifiers=["age"], k_anonymity=k,
        unique_fraction=unique,
    )  # fmt: skip


def _results(rules: dict, passport: Passport, missing: str = "warn") -> dict[str, Verdict]:
    policy = Policy(rules=rules, missing_evidence=missing)
    return {r.name: r.result for r in evaluate(policy, SHA, passport).rules}


def test_privacy_rules() -> None:
    bad = _passport(
        privacy_report=PrivacyReport(
            datasets_scanned=["train"],
            data_findings=[_pii("email"), _pii("email"), _pii("notes")],
            reidentification=[_reid(1, 0.8)],
        )
    )
    rules = {"pii_columns_max": 0, "min_k_anonymity": 5, "unique_record_fraction_max": 0.05}
    assert set(_results(rules, bad).values()) == {Verdict.FAIL}
    result = evaluate(Policy(rules=rules), SHA, bad)
    assert result.rules[0].observed == 2  # distinct columns, not findings
    assert result.verdict is Verdict.FAIL

    good = _passport(
        privacy_report=PrivacyReport(datasets_scanned=["train"], reidentification=[_reid(12, 0.0)])
    )
    assert set(_results(rules, good).values()) == {Verdict.PASS}


def test_warn_and_fail_levels() -> None:
    def leak(auc: float) -> Passport:
        return _passport(
            privacy_report=PrivacyReport(
                leakage=LeakageResult(mia_auc=auc, tpr_at_low_fpr=0.0, members=1, nonmembers=1)
            )
        )

    rules = {"mia_auc_max": {"warn": 0.55, "fail": 0.60}}
    assert _results(rules, leak(0.52))["mia_auc_max"] is Verdict.PASS
    assert _results(rules, leak(0.57))["mia_auc_max"] is Verdict.WARN
    assert _results(rules, leak(0.70))["mia_auc_max"] is Verdict.FAIL


def test_missing_evidence_verdict() -> None:
    rules = {"mia_auc_max": 0.6, "critical_cves_max": 0}
    assert set(_results(rules, _passport()).values()) == {Verdict.WARN}
    assert set(_results(rules, _passport(), missing="fail").values()) == {Verdict.FAIL}


def test_security_rules() -> None:
    report = SecurityReport(
        files_scanned_for_secrets=3,
        secret_findings=[
            Finding(scanner="secrets", category="AWS_ACCESS_KEY_ID", severity=Severity.CRITICAL),
            Finding(scanner="secrets", category="HIGH_ENTROPY_STRING", severity=Severity.MEDIUM),
        ],
        artifacts_scanned=["model.pkl"],
        artifact_findings=[
            Finding(scanner="artifact", category="UNSAFE_PICKLE", severity=Severity.CRITICAL)
        ],
        dependency_audit=DependencyAudit.OK,
        dependency_vulnerabilities=[
            Finding(scanner="pip_audit", category="CVE", severity=Severity.CRITICAL)
        ],
    )
    rules = {"secrets_found_max": 0, "unsafe_pickle": "warn", "critical_cves_max": 0}
    assert _results(rules, _passport(security_report=report)) == {
        "secrets_found_max": Verdict.FAIL,
        "unsafe_pickle": Verdict.WARN,
        "critical_cves_max": Verdict.FAIL,
    }
    clean = SecurityReport(
        files_scanned_for_secrets=3, artifacts_scanned=["m"], dependency_audit=DependencyAudit.OK
    )
    assert set(_results(rules, _passport(security_report=clean)).values()) == {Verdict.PASS}


def test_worst() -> None:
    assert worst([]) is Verdict.PASS
    assert worst([Verdict.PASS, Verdict.WARN]) is Verdict.WARN
    assert worst([Verdict.WARN, Verdict.FAIL, Verdict.PASS]) is Verdict.FAIL


@pytest.mark.parametrize(
    "body",
    [
        "rules: {pii_colums_max: 0}",  # typo
        "rules: {unsafe_pickle: maybe}",
        "rules: {mia_auc_max: {warn: 0.5, oops: 1}}",
        "rules: {min_k_anonymity: five}",
        "rulez: {}",
    ],
)
def test_load_policy_rejects_bad_files(tmp_path: Path, body: str) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(body)
    with pytest.raises(PolicyError):
        load_policy(path)


def test_repo_policy_loads() -> None:
    policy, sha = load_policy(Path(__file__).resolve().parents[1] / "policy.yaml")
    assert "pii_columns_max" in policy.rules
    assert len(sha) == 64
