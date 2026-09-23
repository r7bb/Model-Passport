"""Plain-language summary of a passport, shared by the HTML report and the dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field

from model_passport.core.schema import (
    LeakageResult,
    Passport,
    PrivacyReport,
    SecurityReport,
    Severity,
    Verdict,
    at_least,
)

VERDICT_HEADLINES = {
    Verdict.PASS: "Cleared for its intended use",
    Verdict.WARN: "Usable with caution: some checks need review",
    Verdict.FAIL: "Not cleared: one or more required checks failed",
}
MIN_SAFE_K = 5
MAX_SAFE_ATTACK_AUC = 0.55


@dataclass
class Summary:
    verdict: Verdict | None
    headline: str
    points: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)


def summarize(passport: Passport) -> Summary:
    verdict = passport.policy.verdict if passport.policy else None
    headline = VERDICT_HEADLINES[verdict] if verdict else "No policy was applied to this model"
    summary = Summary(verdict=verdict, headline=headline)
    _describe_provenance(passport, summary)
    if passport.privacy_report is not None:
        _describe_data(passport.privacy_report, summary)
        if passport.privacy_report.leakage is not None:
            _describe_leakage(passport.privacy_report.leakage, summary)
    if passport.security_report is not None:
        _describe_security(passport.security_report, summary)
    drift = [e for e in passport.events if e.event_type == "drift_check"]
    if drift and drift[-1].payload.get("drift_detected"):
        summary.concerns.append("Recent live data no longer looks like the training data.")
    return summary


def _describe_provenance(passport: Passport, summary: Summary) -> None:
    rows = sum(d.row_count or 0 for d in passport.datasets)
    if passport.datasets:
        summary.points.append(
            f"Trained and evaluated on {len(passport.datasets)} dataset(s)"
            + (f" with {rows:,} rows in total." if rows else ".")
        )
    if passport.pipeline:
        names = " → ".join(s.name for s in passport.pipeline)
        summary.points.append(
            f"Built by a recorded {len(passport.pipeline)}-step pipeline: {names}."
        )
    test = passport.metrics.get("test", {})
    if "accuracy" in test:
        summary.points.append(f"Correct on {test['accuracy']:.0%} of held-out test examples.")


def _describe_data(privacy: PrivacyReport, summary: Summary) -> None:
    if not privacy.datasets_scanned:
        return
    direct = {f.location for f in privacy.data_findings if f.details.get("direct_identifier")}
    if direct:
        summary.concerns.append(
            f"Personal identifiers (such as emails or ID numbers) found in {len(direct)} "
            "column(s) of the data."
        )
    else:
        summary.points.append("No personal identifiers were found in the data.")
    ks = [r.k_anonymity for r in privacy.reidentification if r.k_anonymity is not None]
    if not ks:
        return
    k = min(ks)
    if k <= 1:
        summary.concerns.append("Some people in the data can be singled out.")
    elif k < MIN_SAFE_K:
        summary.concerns.append(f"Some people in the data share their profile with only {k - 1}.")
    else:
        summary.points.append(
            f"Every person in the data looks like at least {k} others ({k}-anonymity)."
        )


def _describe_leakage(leakage: LeakageResult, summary: Summary) -> None:
    auc = leakage.mia_auc
    if auc <= MAX_SAFE_ATTACK_AUC:
        summary.points.append(
            "An attacker cannot tell whether someone was in the training data "
            f"(attack success {auc:.2f}, where 0.50 is a coin flip)."
        )
    else:
        summary.concerns.append(
            "The model memorizes its training data: an attacker can guess who was in it "
            f"(attack success {auc:.2f}, where 0.50 is a coin flip)."
        )


def _describe_security(security: SecurityReport, summary: Summary) -> None:
    if any(at_least(f.severity, Severity.HIGH) for f in security.secret_findings):
        summary.concerns.append("Passwords or API keys were found in the data or code.")
    if any(f.category == "UNSAFE_PICKLE" for f in security.artifact_findings):
        summary.concerns.append("The model file contains code that runs when it is opened.")
    critical = [f for f in security.dependency_vulnerabilities if f.severity is Severity.CRITICAL]
    if critical:
        summary.concerns.append(f"{len(critical)} critical security flaw(s) in its software.")
