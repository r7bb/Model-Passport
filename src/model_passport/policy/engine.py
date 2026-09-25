"""Policy engine: evaluate ``policy.yaml`` rules against a passport's reports.

Threshold forms::

    min_k_anonymity: 5                  # fail below 5
    mia_auc_max: {warn: 0.55, fail: 0.60}
    unsafe_pickle: warn                 # verdict to apply when the condition is met

A rule whose evidence was never collected (e.g. no leakage audit ran) gets the
``missing_evidence`` verdict (default ``warn``), never a silent pass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from model_passport.core import identity
from model_passport.core.schema import (
    EntityAuditResult,
    Passport,
    PolicyResult,
    ReidentificationResult,
    RuleResult,
    Severity,
    Verdict,
    at_least,
)

VERDICT_ORDER = [Verdict.PASS, Verdict.WARN, Verdict.FAIL]
# A pickle we cannot parse is as untrustworthy as one importing os.system.
UNSAFE_ARTIFACT_CATEGORIES = {"UNSAFE_PICKLE", "SCAN_ERROR"}


class PolicyError(Exception):
    """Raised for malformed policies or unknown rules."""


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: dict[str, Any] = Field(default_factory=dict)
    missing_evidence: Verdict = Verdict.WARN


def worst(verdicts: list[Verdict]) -> Verdict:
    return max(verdicts, key=VERDICT_ORDER.index, default=Verdict.PASS)


def load_policy(path: Path) -> tuple[Policy, str]:
    """Load a policy file; returns the policy and the file's SHA256."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        policy = Policy.model_validate(data)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise PolicyError(f"cannot load policy {path}: {exc}") from exc
    unknown = sorted(set(policy.rules) - set(RULES))
    if unknown:
        raise PolicyError(f"unknown policy rule(s): {', '.join(unknown)}")
    for name, threshold in policy.rules.items():
        if RULES[name].kind == "flag":
            if threshold not in {v.value for v in Verdict}:
                raise PolicyError(f"rule {name}: expected pass, warn, or fail")
        else:
            _thresholds(name, threshold)
    return policy, identity.sha256_file(path)


# --- Evidence extraction -------------------------------------------------------------------

Missing = None  # sentinel: evidence not collected


def _pii_columns(p: Passport) -> int | None:
    report = p.privacy_report
    if report is None or not report.datasets_scanned:
        return Missing
    return len(
        {
            f.location
            for f in report.data_findings
            if f.scanner == "pii" and f.details.get("direct_identifier")
        }
    )


def _reid_results(p: Passport) -> list[ReidentificationResult]:
    return p.privacy_report.reidentification if p.privacy_report else []


def _min_k(p: Passport) -> int | None:
    values = [r.k_anonymity for r in _reid_results(p) if r.k_anonymity is not None]
    return min(values) if values else Missing


def _max_unique(p: Passport) -> float | None:
    results = [r for r in _reid_results(p) if r.quasi_identifiers]
    return max(r.unique_fraction for r in results) if results else Missing


def _min_l(p: Passport) -> int | None:
    values = [r.l_diversity for r in _reid_results(p) if r.l_diversity is not None]
    return min(values) if values else Missing


def _leakage(field: str) -> Callable[[Passport], float | None]:
    """Evidence extractor for one LeakageResult field."""

    def extract(p: Passport) -> float | None:
        leakage = p.privacy_report.leakage if p.privacy_report else None
        return getattr(leakage, field) if leakage else Missing

    return extract


def _secrets(p: Passport) -> int | None:
    report = p.security_report
    if report is None or report.files_scanned_for_secrets is None:
        return Missing
    return sum(1 for f in report.secret_findings if at_least(f.severity, Severity.HIGH))


def _unsafe_pickles(p: Passport) -> int | None:
    report = p.security_report
    if report is None or not report.artifacts_scanned:
        return Missing
    return sum(1 for f in report.artifact_findings if f.category in UNSAFE_ARTIFACT_CATEGORIES)


def _cves(severity: Severity) -> Callable[[Passport], int | None]:
    def count(p: Passport) -> int | None:
        report = p.security_report
        if report is None or report.dependency_audit != "ok":
            return Missing
        return sum(1 for f in report.dependency_vulnerabilities if f.severity == severity)

    return count


def _entity_audit(p: Passport) -> EntityAuditResult | None:
    return p.privacy_report.entity_audit if p.privacy_report else None


def _entities_at(severity: Severity) -> Callable[[Passport], int | None]:
    """Entities rated exactly ``severity`` by the entity-level audit."""

    def count(p: Passport) -> int | None:
        audit = _entity_audit(p)
        return Missing if audit is None else audit.severity_counts.get(severity.value, 0)

    return count


def _entity_auc(p: Passport) -> float | None:
    audit = _entity_audit(p)
    return Missing if audit is None else audit.auc


def _entity_tpr(p: Passport) -> float | None:
    audit = _entity_audit(p)
    return Missing if audit is None else audit.tpr_at_fpr.get("0.01")


# --- Rules ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    evidence: Callable[[Passport], Any]
    kind: str  # "max": observed must be <= threshold; "min": >=; "flag": verdict if observed > 0
    description: str
    hint: str = ""  # how to collect the evidence when it is missing


QUASI_HINT = "no quasi-identifiers found; list them under privacy.quasi_identifiers"
SENSITIVE_HINT = "set privacy.sensitive_column"
LEAKAGE_HINT = "set audit.label_column and declare train and test datasets"
ENTITY_HINT = "run `passport llm audit` and set privacy.entity_audit"


RULES: dict[str, Rule] = {
    "pii_columns_max": Rule(_pii_columns, "max", "columns containing direct identifiers"),
    "min_k_anonymity": Rule(_min_k, "min", "k-anonymity over quasi-identifiers", QUASI_HINT),
    "unique_record_fraction_max": Rule(
        _max_unique, "max", "fraction of unique records", QUASI_HINT
    ),
    "min_l_diversity": Rule(_min_l, "min", "l-diversity of the sensitive column", SENSITIVE_HINT),
    "mia_auc_max": Rule(
        _leakage("mia_auc"), "max", "membership inference attack AUC", LEAKAGE_HINT
    ),
    "mia_tpr_at_low_fpr_max": Rule(
        _leakage("tpr_at_low_fpr"), "max", "attack TPR at low FPR", LEAKAGE_HINT
    ),
    "generalization_gap_max": Rule(
        _leakage("generalization_gap"), "max", "train minus test metric", LEAKAGE_HINT
    ),
    "secrets_found_max": Rule(_secrets, "max", "confirmed secrets in data and scripts"),
    "unsafe_pickle": Rule(_unsafe_pickles, "flag", "pickle files with dangerous imports"),
    "critical_cves_max": Rule(_cves(Severity.CRITICAL), "max", "critical dependency CVEs"),
    "entity_critical_max": Rule(
        _entities_at(Severity.CRITICAL), "max", "critical memorized entities", ENTITY_HINT
    ),
    "entity_high_max": Rule(
        _entities_at(Severity.HIGH), "max", "high-risk memorized entities", ENTITY_HINT
    ),
    "el_mia_auc_max": Rule(_entity_auc, "max", "entity-level attack AUC", ENTITY_HINT),
    "el_mia_tpr_at_1pct_fpr_max": Rule(
        _entity_tpr, "max", "entity-level attack TPR at 1% FPR", ENTITY_HINT
    ),
    "high_cves_max": Rule(_cves(Severity.HIGH), "max", "high severity dependency CVEs"),
}


def _thresholds(name: str, value: Any) -> tuple[float | None, float | None]:
    """Normalize a threshold to (warn, fail)."""
    if isinstance(value, dict):
        extra = set(value) - {"warn", "fail"}
        if extra or not value:
            raise PolicyError(f"rule {name}: expected keys warn and/or fail, got {sorted(value)}")
        return value.get("warn"), value.get("fail")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PolicyError(f"rule {name}: threshold must be a number or {{warn, fail}}")
    return None, value


def _violates(kind: str, observed: float, threshold: float | None) -> bool:
    if threshold is None:
        return False
    return observed > threshold if kind == "max" else observed < threshold


def evaluate_rule(name: str, threshold: Any, passport: Passport, missing: Verdict) -> RuleResult:
    rule = RULES[name]
    observed = rule.evidence(passport)
    if observed is Missing:
        return RuleResult(
            name=name, threshold=threshold, observed=None, result=missing,
            message=f"not evaluated: no evidence for {rule.description}"
            + (f" ({rule.hint})" if rule.hint else ""),
        )  # fmt: skip

    if rule.kind == "flag":
        result = Verdict(threshold) if observed > 0 else Verdict.PASS
        message = f"{observed} {rule.description}" if observed else f"no {rule.description}"
        return RuleResult(
            name=name, threshold=threshold, observed=observed, result=result, message=message
        )

    warn_at, fail_at = _thresholds(name, threshold)
    if _violates(rule.kind, observed, fail_at):
        result = Verdict.FAIL
    elif _violates(rule.kind, observed, warn_at):
        result = Verdict.WARN
    else:
        result = Verdict.PASS
    comparison = "<=" if rule.kind == "max" else ">="
    limit = fail_at if fail_at is not None else warn_at
    shown = round(observed, 4) if isinstance(observed, float) else observed
    return RuleResult(
        name=name,
        threshold=threshold,
        observed=shown,
        result=result,
        message=f"{rule.description}: {shown} (required {comparison} {limit})",
    )


def evaluate(policy: Policy, policy_sha256: str, passport: Passport) -> PolicyResult:
    results = [
        evaluate_rule(name, threshold, passport, policy.missing_evidence)
        for name, threshold in policy.rules.items()
    ]
    return PolicyResult(
        policy_sha256=policy_sha256,
        rules=results,
        verdict=worst([r.result for r in results]),
    )
