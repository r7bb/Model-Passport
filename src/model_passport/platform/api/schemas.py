"""Request and response bodies of the platform API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from model_passport.platform.models import (
    Access,
    JobStatus,
    Plan,
    Role,
    State,
    TenantStatus,
)


class _Out(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Login(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 - the OAuth token type, not a secret
    expires_in_minutes: int


class MembershipOut(BaseModel):
    tenant: str
    role: Role


class Me(BaseModel):
    id: str
    email: str
    name: str
    super_admin: bool
    memberships: list[MembershipOut]


class TenantIn(BaseModel):
    slug: str
    name: str
    plan: Plan = Plan.FREE


class TenantPatch(BaseModel):
    plan: Plan | None = None
    status: TenantStatus | None = None


class TenantOut(_Out):
    id: str
    slug: str
    name: str
    plan: Plan
    status: TenantStatus
    public_key: str | None = None
    created_at: datetime


class MemberIn(BaseModel):
    email: str
    role: Role
    name: str = ""
    password: str | None = Field(
        default=None, description="Needed only when the person has no account yet."
    )


class MemberPatch(BaseModel):
    role: Role


class MemberOut(BaseModel):
    user_id: str
    email: str
    name: str
    role: Role


class DatasetOut(_Out):
    id: str
    name: str
    source: str
    license: str
    consent: str
    sha256: str
    size_bytes: int
    records: int
    entities: int
    created_at: datetime


class ModelIn(BaseModel):
    name: str
    access: Access
    base: str = Field(description="hf:<id>, a checkpoint, anthropic:<model>, openai:<model>, tiny")
    description: str = ""


class VersionIn(BaseModel):
    version: str = "1.0.0"
    dataset_id: str | None = Field(default=None, description="The data it was trained on.")
    reference_dataset_id: str = Field(description="The original corpus audits test against.")
    train: bool = Field(default=False, description="Fine-tune on the platform, then audit.")


class VersionOut(_Out):
    id: str
    model_id: str
    version: str
    state: State
    parent_id: str | None
    dataset_id: str | None
    reference_dataset_id: str | None
    artifact_sha256: str | None
    verdict: str | None
    auc: float | None
    critical: int
    high: int
    attestation_sha256: str | None
    created_at: datetime
    updated_at: datetime


class ModelOut(_Out):
    id: str
    name: str
    access: Access
    base: str
    description: str
    created_at: datetime


class ModelDetail(ModelOut):
    versions: list[VersionOut]


class TransitionIn(BaseModel):
    to: State
    reason: str = ""


class DecisionIn(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")
    comment: str = ""


class KillIn(BaseModel):
    reason: str = Field(min_length=3)


class JobOut(_Out):
    id: str
    kind: str
    status: JobStatus
    payload: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    attempts: int
    created_at: datetime
    finished_at: datetime | None


class EventOut(_Out):
    seq: int
    actor: str
    action: str
    target_type: str
    target_id: str
    details: dict[str, Any]
    hash: str
    created_at: datetime


class ChainStatus(BaseModel):
    events: int
    intact: bool
    problems: list[str]
