"""The model version lifecycle as an explicit state machine (the product flow).

::

    PHASE 1 DETECT        registered -> auditing -> findings | clean
    PHASE 2 REMEDIATE     findings -> remediating -> superseded   (a new version is registered)
    PHASE 3 VERIFY        clean -> canary -> verifying -> approved -> released
                          verifying -> findings        (risk not below threshold: remediate again)
                          verifying | approved -> rejected
    OPERATE               released -> rolled_back
                          any pre-release or live state -> killed   (the kill switch)

Each transition needs a permission, and every change is written to the audit log in the same
transaction.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from model_passport.platform import auditlog
from model_passport.platform.models import ModelVersion, Role, State
from model_passport.platform.rbac import Permission, allowed

S, P = State, Permission
SYSTEM_ONLY = None  # transitions made by workers after a job, never by a person

TRANSITIONS: dict[tuple[State, State], Permission | None] = {
    (S.REGISTERED, S.AUDITING): P.AUDITS_RUN,
    (S.CLEAN, S.AUDITING): P.AUDITS_RUN,  # re-audit, e.g. after new data
    (S.FINDINGS, S.AUDITING): P.AUDITS_RUN,
    (S.AUDITING, S.FINDINGS): SYSTEM_ONLY,
    (S.AUDITING, S.CLEAN): SYSTEM_ONLY,
    (S.AUDITING, S.REGISTERED): SYSTEM_ONLY,  # the audit job failed; it can be retried
    (S.FINDINGS, S.REMEDIATING): P.REMEDIATION_RUN,
    (S.REMEDIATING, S.SUPERSEDED): SYSTEM_ONLY,
    (S.REMEDIATING, S.FINDINGS): SYSTEM_ONLY,  # the remediation job failed
    (S.CLEAN, S.CANARY): P.DEV_ENDPOINTS,
    (S.CANARY, S.VERIFYING): P.DEV_ENDPOINTS,
    (S.VERIFYING, S.FINDINGS): P.APPROVALS_DECIDE,  # risk not below threshold
    (S.VERIFYING, S.APPROVED): P.APPROVALS_DECIDE,
    (S.VERIFYING, S.REJECTED): P.APPROVALS_DECIDE,
    (S.APPROVED, S.REJECTED): P.APPROVALS_DECIDE,
    (S.APPROVED, S.RELEASED): P.RELEASES_MANAGE,
    (S.RELEASED, S.ROLLED_BACK): P.RELEASES_MANAGE,
}
KILLABLE = {S.FINDINGS, S.CLEAN, S.CANARY, S.VERIFYING, S.APPROVED, S.RELEASED}
for _state in KILLABLE:
    TRANSITIONS[(_state, S.KILLED)] = P.KILL_SWITCH


class TransitionError(ValueError):
    """The transition is not part of the lifecycle, or the actor may not make it."""


@dataclass(frozen=True)
class Who:
    actor: auditlog.Actor
    role: Role | None
    super_admin: bool = False
    system: bool = False


def next_states(state: State) -> list[State]:
    return sorted({to for (frm, to) in TRANSITIONS if frm is state}, key=lambda s: s.value)


def check(version: ModelVersion, to: State, who: Who) -> None:
    """Raise ``TransitionError`` unless ``who`` may move ``version`` to ``to`` now."""
    key = (version.state, to)
    if key not in TRANSITIONS:
        raise TransitionError(f"{version.state.value} -> {to.value} is not a lifecycle step")
    needed = TRANSITIONS[key]
    if needed is SYSTEM_ONLY:
        if not who.system:
            raise TransitionError(f"{to.value} is set by the platform after a job, not by hand")
    elif not who.system and not allowed(who.role, needed, who.super_admin):
        raise TransitionError(f"{needed.value} is required for {version.state.value} -> {to.value}")


def move(session: Session, version: ModelVersion, to: State, who: Who, reason: str = "") -> None:
    """Change ``version`` to ``to`` if the lifecycle and the actor's role allow it."""
    check(version, to, who)
    before = version.state
    version.state = to
    auditlog.record(
        session, version.tenant_id, who.actor, f"version.{to.value}", "model_version",
        version.id, {"from": before.value, "to": to.value, "reason": reason},
    )  # fmt: skip
