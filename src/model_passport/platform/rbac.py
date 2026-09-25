"""Who may do what (M2 Access & Roles), following the roles in the product flow.

| Role | Purpose |
|---|---|
| Super Admin | manages all organizations and subscriptions (platform-wide) |
| Org Admin | manages users, roles, approvals, and releases |
| ML Engineer | registers, audits, and retrains models |
| Compliance Auditor | reviews findings, approves releases, can activate the kill switch |
| Canary Tester | tries to extract leaked data from developer endpoints |
| End Consumer | uses the approved model |
| External Reviewer | an investor or acquirer who receives the diligence report |
"""

from __future__ import annotations

from enum import StrEnum

from model_passport.platform.models import Role


class Permission(StrEnum):
    TENANTS_MANAGE = "tenants.manage"  # super admin only
    MEMBERS_MANAGE = "members.manage"
    MODELS_READ = "models.read"
    MODELS_WRITE = "models.write"  # register models, versions, and datasets
    AUDITS_RUN = "audits.run"
    REMEDIATION_RUN = "remediation.run"
    FINDINGS_READ = "findings.read"
    APPROVALS_DECIDE = "approvals.decide"
    RELEASES_MANAGE = "releases.manage"
    KILL_SWITCH = "killswitch.activate"
    DEV_ENDPOINTS = "endpoints.dev"  # deploy to and test developer endpoints
    TEST_REPORTS = "testreports.submit"
    CONSUMER_USE = "endpoints.consumer"
    AUDIT_LOG_READ = "auditlog.read"
    ANALYTICS_READ = "analytics.read"
    PROVENANCE_READ = "provenance.read"
    REPORTS_READ = "reports.read"
    JOBS_READ = "jobs.read"


P = Permission
_READ = {P.MODELS_READ, P.FINDINGS_READ, P.ANALYTICS_READ, P.PROVENANCE_READ, P.JOBS_READ}

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.ORG_ADMIN: frozenset(
        _READ
        | {
            P.MEMBERS_MANAGE,
            P.APPROVALS_DECIDE,
            P.RELEASES_MANAGE,
            P.KILL_SWITCH,
            P.AUDIT_LOG_READ,
            P.REPORTS_READ,
        }
    ),
    Role.ML_ENGINEER: frozenset(
        _READ | {P.MODELS_WRITE, P.AUDITS_RUN, P.REMEDIATION_RUN, P.DEV_ENDPOINTS}
    ),
    Role.COMPLIANCE_AUDITOR: frozenset(
        _READ | {P.APPROVALS_DECIDE, P.KILL_SWITCH, P.AUDIT_LOG_READ, P.REPORTS_READ}
    ),
    Role.CANARY_TESTER: frozenset({P.MODELS_READ, P.DEV_ENDPOINTS, P.TEST_REPORTS}),
    Role.END_CONSUMER: frozenset({P.CONSUMER_USE}),
    Role.EXTERNAL_REVIEWER: frozenset({P.REPORTS_READ, P.PROVENANCE_READ}),
}


def allowed(role: Role | None, permission: Permission, super_admin: bool = False) -> bool:
    """Super admins may do anything; others need a role that grants ``permission``."""
    if super_admin:
        return True
    return role is not None and permission in ROLE_PERMISSIONS[role]
