"""The release gate for platform versions: the same policy engine as ``passport build``.

Tenants start from the industry-based language model policy (``llm.project.POLICY_TEMPLATE``):
confirmed Critical and High findings block release, and the attack's AUC and TPR at 1% FPR must
stay near chance. Only the entity-audit rules apply here; the rest of a project policy (secrets,
CVEs) is checked when a full passport is built.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import cast

import yaml

from model_passport.core.schema import (
    EntityAuditResult,
    Passport,
    PolicyResult,
    PrivacyReport,
    Verdict,
)
from model_passport.llm.project import POLICY_TEMPLATE
from model_passport.policy.engine import Policy, evaluate

ENTITY_RULES = {
    "entity_critical_max",
    "entity_high_max",
    "el_mia_auc_max",
    "el_mia_tpr_at_1pct_fpr_max",
}


def default_policy() -> tuple[Policy, str]:
    data = yaml.safe_load(POLICY_TEMPLATE)
    rules = {k: v for k, v in data["rules"].items() if k in ENTITY_RULES}
    policy = Policy(rules=rules, missing_evidence=Verdict(data["missing_evidence"]))
    return policy, hashlib.sha256(POLICY_TEMPLATE.encode()).hexdigest()


def check(audit: EntityAuditResult, policy: tuple[Policy, str] | None = None) -> PolicyResult:
    """The gate's verdict for one audit."""
    rules, digest = policy or default_policy()
    # The entity rules read only the privacy and security reports of their evidence.
    evidence = SimpleNamespace(
        privacy_report=PrivacyReport(entity_audit=audit), security_report=None
    )
    return evaluate(rules, digest, cast(Passport, evidence))


def blocking(result: PolicyResult) -> bool:
    return result.verdict is Verdict.FAIL
