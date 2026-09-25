"""Platform data model (SQLAlchemy 2.0, portable across PostgreSQL and SQLite).

Every tenant-owned row carries ``tenant_id``. Queries go through ``TenantScope``, and on
PostgreSQL row-level security enforces the same boundary inside the database (see the initial
migration).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return str(uuid.uuid4())


def now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON, list[Any]: JSON}


def _enum(kind: type[StrEnum]) -> SqlEnum:
    return SqlEnum(
        kind, native_enum=False, length=32, values_callable=lambda e: [m.value for m in e]
    )


class Role(StrEnum):
    """Roles within a tenant. Super admins are platform-wide (``User.is_super_admin``)."""

    ORG_ADMIN = "org_admin"
    ML_ENGINEER = "ml_engineer"
    COMPLIANCE_AUDITOR = "compliance_auditor"
    CANARY_TESTER = "canary_tester"
    END_CONSUMER = "end_consumer"
    EXTERNAL_REVIEWER = "external_reviewer"


class Plan(StrEnum):
    FREE = "free"
    TEAM = "team"
    ENTERPRISE = "enterprise"


class TenantStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


class Access(StrEnum):
    OPEN_WEIGHT = "open-weight"
    API = "api"


class State(StrEnum):
    """Lifecycle of a model version, following the product flow."""

    REGISTERED = "registered"
    AUDITING = "auditing"
    FINDINGS = "findings"  # high-risk entities found: remediate
    CLEAN = "clean"  # audit passed: may go to canary
    REMEDIATING = "remediating"
    SUPERSEDED = "superseded"  # a remediated version replaced it
    CANARY = "canary"  # deployed to developer endpoints only
    VERIFYING = "verifying"
    APPROVED = "approved"
    RELEASED = "released"  # live for consumers, monitored
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"
    KILLED = "killed"  # kill switch: blocked everywhere


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Environment(StrEnum):
    DEV = "dev"  # developer endpoints for canary testing
    CONSUMER = "consumer"


class DeploymentStatus(StrEnum):
    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"
    KILLED = "killed"


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    slug: Mapped[str] = mapped_column(String(63), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    plan: Mapped[Plan] = mapped_column(_enum(Plan), default=Plan.FREE)
    status: Mapped[TenantStatus] = mapped_column(_enum(TenantStatus), default=TenantStatus.ACTIVE)
    wrapped_key: Mapped[bytes] = mapped_column(LargeBinary, doc="Tenant data key, wrapped.")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    is_super_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    memberships: Mapped[list[Membership]] = relationship(back_populates="user")


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("user_id", "tenant_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    role: Mapped[Role] = mapped_column(_enum(Role))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    user: Mapped[User] = relationship(back_populates="memberships")


CONSENT_DOC = "obtained, not-required, or unknown"


class Dataset(Base):
    """A training dataset and where it came from (M9 Data Provenance)."""

    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("tenant_id", "name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(Text, default="")
    license: Mapped[str] = mapped_column(String(200), default="unknown")
    consent: Mapped[str] = mapped_column(String(32), default="unknown", doc=CONSENT_DOC)
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    object_key: Mapped[str] = mapped_column(String(500))
    records: Mapped[int] = mapped_column(Integer, default=0)
    entities: Mapped[int] = mapped_column(Integer, default=0)
    created_by: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Model(Base):
    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("tenant_id", "name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    access: Mapped[Access] = mapped_column(_enum(Access))
    base: Mapped[str] = mapped_column(String(500), doc="Model spec: hf:<id>, anthropic:<model>...")
    created_by: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    versions: Mapped[list[ModelVersion]] = relationship(
        back_populates="model", order_by="ModelVersion.created_at"
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("model_id", "version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"))
    version: Mapped[str] = mapped_column(String(64))
    state: Mapped[State] = mapped_column(_enum(State), default=State.REGISTERED)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("model_versions.id"))
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id"))
    reference_dataset_id: Mapped[str | None] = mapped_column(
        ForeignKey("datasets.id"), doc="Original corpus the audit tests against."
    )
    artifact_key: Mapped[str | None] = mapped_column(String(500))
    artifact_sha256: Mapped[str | None] = mapped_column(String(64))
    audit: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    verdict: Mapped[str | None] = mapped_column(String(16))
    auc: Mapped[float | None] = mapped_column(Float)
    critical: Mapped[int] = mapped_column(Integer, default=0)
    high: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    model: Mapped[Model] = relationship(back_populates="versions")


class Approval(Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    role: Mapped[Role] = mapped_column(_enum(Role))
    decision: Mapped[str] = mapped_column(String(16))  # approve | reject
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"))
    environment: Mapped[Environment] = mapped_column(_enum(Environment))
    status: Mapped[DeploymentStatus] = mapped_column(
        _enum(DeploymentStatus), default=DeploymentStatus.ACTIVE
    )
    endpoint: Mapped[str] = mapped_column(String(500), default="")
    created_by: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class TestReport(Base):
    """A canary tester's extraction attempt against a developer endpoint."""

    __tablename__ = "test_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    prompt_count: Mapped[int] = mapped_column(Integer, default=0)
    leaks_found: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Job(Base):
    """Background work for workers (audit, remediate, report), claimed with row locks."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[JobStatus] = mapped_column(_enum(JobStatus), default=JobStatus.QUEUED)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    locked_by: Mapped[str | None] = mapped_column(String(100))
    created_by: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditEvent(Base):
    """Append-only, hash-chained record of every action (M4 Audit Logs).

    ``hash`` covers the event's fields and the previous event's hash in the same tenant, so an
    edit, deletion, or reordering breaks the chain. Database triggers reject UPDATE and DELETE.
    """

    __tablename__ = "audit_events"
    __table_args__ = (UniqueConstraint("tenant_id", "seq"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(36), doc="'platform' for platform-wide events.")
    seq: Mapped[int] = mapped_column(Integer)
    actor_id: Mapped[str | None] = mapped_column(String(36))
    actor: Mapped[str] = mapped_column(String(320), default="system")
    action: Mapped[str] = mapped_column(String(64))
    target_type: Mapped[str] = mapped_column(String(32), default="")
    target_id: Mapped[str] = mapped_column(String(64), default="")
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


TENANT_TABLES = (
    "memberships", "datasets", "models", "model_versions", "approvals", "deployments",
    "test_reports", "jobs",
)  # fmt: skip
