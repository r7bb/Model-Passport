"""The job queue (M7 Pipelines & Workers), stored in the database.

Workers claim jobs with ``SELECT ... FOR UPDATE SKIP LOCKED`` on PostgreSQL, so any number can
run side by side without taking the same job; SQLite processes one at a time. A failed job is
retried until ``max_attempts``, and every state change goes to the audit log.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from model_passport.platform import auditlog
from model_passport.platform.models import Job, JobStatus, now

KINDS = ("audit", "remediate", "report")


class JobError(ValueError):
    """The job cannot be queued or updated."""


def enqueue(
    session: Session,
    tenant_id: str,
    kind: str,
    payload: dict[str, Any],
    actor: auditlog.Actor,
    max_attempts: int = 3,
) -> Job:
    if kind not in KINDS:
        raise JobError(f"unknown job kind {kind!r}; expected one of {', '.join(KINDS)}")
    job = Job(
        tenant_id=tenant_id, kind=kind, payload=payload, max_attempts=max_attempts,
        created_by=actor.id,
    )  # fmt: skip
    session.add(job)
    session.flush()
    auditlog.record(session, tenant_id, actor, "job.queued", "job", job.id, {"kind": kind})
    return job


def claim(session: Session, worker: str, kinds: tuple[str, ...] = KINDS) -> Job | None:
    """The oldest queued job, marked running for ``worker``; None when the queue is empty."""
    query = (
        select(Job)
        .where(Job.status == JobStatus.QUEUED, Job.kind.in_(kinds))
        .order_by(Job.created_at)
        .limit(1)
    )
    if session.get_bind().dialect.name == "postgresql":
        query = query.with_for_update(skip_locked=True)
    job = session.scalars(query).first()
    if job is None:
        return None
    job.status = JobStatus.RUNNING
    job.attempts += 1
    job.locked_by = worker
    job.started_at = now()
    auditlog.record(
        session, job.tenant_id, auditlog.Actor(None, f"worker:{worker}"), "job.started", "job",
        job.id, {"kind": job.kind, "attempt": job.attempts},
    )  # fmt: skip
    return job


def finish(session: Session, job: Job, result: dict[str, Any]) -> None:
    job.status = JobStatus.SUCCEEDED
    job.result = result
    job.error = None
    job.finished_at = now()
    auditlog.record(
        session, job.tenant_id, auditlog.Actor(None, f"worker:{job.locked_by}"),
        "job.succeeded", "job", job.id, {"kind": job.kind},
    )  # fmt: skip


def fail(session: Session, job: Job, error: str) -> None:
    """Record a failure; the job is queued again unless it is out of attempts."""
    retry = job.attempts < job.max_attempts
    job.status = JobStatus.QUEUED if retry else JobStatus.FAILED
    job.error = error[:2000]
    job.finished_at = None if retry else now()
    auditlog.record(
        session, job.tenant_id, auditlog.Actor(None, f"worker:{job.locked_by}"),
        "job.retrying" if retry else "job.failed", "job", job.id,
        {"kind": job.kind, "attempt": job.attempts, "error": error[:200]},
    )  # fmt: skip
