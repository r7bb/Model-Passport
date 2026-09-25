"""The append-only, hash-chained audit log (M4). Every lifecycle step is recorded here.

Each event's hash covers its fields and the previous event's hash in the same tenant (the
first event chains from a fixed genesis value). ``verify_chain`` recomputes every hash, so an
edited, deleted, inserted, or reordered event is detected even by someone with database access
who bypasses the triggers that reject UPDATE and DELETE.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from model_passport.platform.models import AuditEvent

PLATFORM = "platform"  # tenant id for platform-wide events (tenants, subscriptions)
GENESIS = "0" * 64


@dataclass(frozen=True)
class Actor:
    id: str | None
    label: str  # email, or "system" / "worker:<name>"


SYSTEM = Actor(None, "system")


def _digest(event: AuditEvent) -> str:
    body = {
        "tenant": event.tenant_id,
        "seq": event.seq,
        "actor_id": event.actor_id,
        "actor": event.actor,
        "action": event.action,
        "target_type": event.target_type,
        "target_id": event.target_id,
        "details": event.details,
        "prev": event.prev_hash,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def record(
    session: Session,
    tenant_id: str,
    actor: Actor,
    action: str,
    target_type: str = "",
    target_id: str = "",
    details: dict[str, Any] | None = None,
) -> AuditEvent:
    """Append an event in the caller's transaction (it commits or rolls back with the action).

    On PostgreSQL the tenant's chain is locked for the transaction, so concurrent writers
    cannot fork it.
    """
    if session.get_bind().dialect.name == "postgresql":
        session.execute(select(func.pg_advisory_xact_lock(func.hashtext(tenant_id))))
    last = session.scalars(
        select(AuditEvent)
        .where(AuditEvent.tenant_id == tenant_id)
        .order_by(AuditEvent.seq.desc())
        .limit(1)
    ).first()
    event = AuditEvent(
        tenant_id=tenant_id,
        seq=(last.seq + 1) if last else 1,
        actor_id=actor.id,
        actor=actor.label,
        action=action,
        target_type=target_type,
        target_id=target_id,
        details=details or {},
        prev_hash=last.hash if last else GENESIS,
    )
    event.hash = _digest(event)
    session.add(event)
    session.flush()
    return event


def verify_chain(events: Sequence[AuditEvent]) -> list[str]:
    """Problems in a tenant's chain, in order (empty when intact)."""
    problems = []
    expected_prev, expected_seq = GENESIS, 1
    for event in events:
        if event.seq != expected_seq:
            problems.append(f"event {event.seq}: expected sequence {expected_seq}")
        if event.prev_hash != expected_prev:
            problems.append(f"event {event.seq}: does not chain from the previous event")
        if _digest(event) != event.hash:
            problems.append(f"event {event.seq}: contents changed since it was recorded")
        expected_prev, expected_seq = event.hash, event.seq + 1
    return problems


def chain(session: Session, tenant_id: str) -> list[AuditEvent]:
    return list(
        session.scalars(
            select(AuditEvent).where(AuditEvent.tenant_id == tenant_id).order_by(AuditEvent.seq)
        )
    )
