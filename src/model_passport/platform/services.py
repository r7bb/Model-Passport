"""Platform operations shared by the API, the workers, and the CLI.

Every function takes the actor making the change and records it in the audit log in the same
transaction, so the log and the data can never disagree.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from model_passport.platform import auditlog
from model_passport.platform.auditlog import Actor
from model_passport.platform.models import (
    Access,
    Dataset,
    Membership,
    Model,
    ModelVersion,
    Plan,
    Role,
    Tenant,
    TenantStatus,
    User,
)
from model_passport.platform.security import hash_password, new_data_key, unwrap_key, wrap_key
from model_passport.platform.storage import ObjectStore, TenantStore

SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
RESERVED_SLUGS = {"www", "api", "admin", "app", "platform", "static", "status"}
MIN_PASSWORD = 12


class ServiceError(ValueError):
    """The request is invalid; the message is safe to show to the user."""


class NotFoundError(ServiceError):
    """The object does not exist in this tenant."""


# --- Tenants and people --------------------------------------------------------------------


def create_tenant(
    session: Session, master_key: bytes, slug: str, name: str, actor: Actor, plan: Plan = Plan.FREE
) -> Tenant:
    """A new organization with its own data key (subdomain ``<slug>.<base domain>``)."""
    if not SLUG_RE.fullmatch(slug) or slug in RESERVED_SLUGS:
        raise ServiceError("slug must be 1-63 lowercase letters, digits, or hyphens (not reserved)")
    if session.scalars(select(Tenant).where(Tenant.slug == slug)).first():
        raise ServiceError(f"tenant {slug!r} already exists")
    tenant = Tenant(slug=slug, name=name, plan=plan, wrapped_key=b"")
    session.add(tenant)
    session.flush()
    tenant.wrapped_key = wrap_key(master_key, new_data_key(), tenant.id)
    auditlog.record(session, auditlog.PLATFORM, actor, "tenant.created", "tenant", tenant.id,
                    {"slug": slug, "plan": plan.value})  # fmt: skip
    return tenant


def update_tenant(
    session: Session, tenant: Tenant, actor: Actor, plan: Plan | None, status: TenantStatus | None
) -> Tenant:
    changes = {}
    if plan is not None and plan is not tenant.plan:
        tenant.plan, changes["plan"] = plan, plan.value
    if status is not None and status is not tenant.status:
        tenant.status, changes["status"] = status, status.value
    if changes:
        auditlog.record(
            session, auditlog.PLATFORM, actor, "tenant.updated", "tenant", tenant.id, changes
        )
    return tenant


def create_user(
    session: Session, email: str, password: str, name: str = "", super_admin: bool = False
) -> User:
    email = email.strip().lower()
    if "@" not in email:
        raise ServiceError("a valid email address is required")
    if len(password) < MIN_PASSWORD:
        raise ServiceError(f"passwords need at least {MIN_PASSWORD} characters")
    if session.scalars(select(User).where(User.email == email)).first():
        raise ServiceError(f"{email} already has an account")
    user = User(email=email, name=name, password_hash=hash_password(password),
                is_super_admin=super_admin)  # fmt: skip
    session.add(user)
    session.flush()
    return user


def add_member(
    session: Session, tenant: Tenant, user: User, role: Role, actor: Actor
) -> Membership:
    existing = session.scalars(
        select(Membership).where(Membership.tenant_id == tenant.id, Membership.user_id == user.id)
    ).first()
    if existing is not None:
        before = existing.role
        existing.role = role
        auditlog.record(session, tenant.id, actor, "member.role_changed", "user", user.id,
                        {"from": before.value, "to": role.value})  # fmt: skip
        return existing
    membership = Membership(tenant_id=tenant.id, user_id=user.id, role=role)
    session.add(membership)
    session.flush()
    auditlog.record(session, tenant.id, actor, "member.added", "user", user.id,
                    {"email": user.email, "role": role.value})  # fmt: skip
    return membership


def remove_member(session: Session, membership: Membership, actor: Actor) -> None:
    auditlog.record(session, membership.tenant_id, actor, "member.removed", "user",
                    membership.user_id, {"role": membership.role.value})  # fmt: skip
    session.delete(membership)


def tenant_store(store: ObjectStore, master_key: bytes, tenant: Tenant) -> TenantStore:
    return TenantStore(store, tenant.id, unwrap_key(master_key, tenant.wrapped_key, tenant.id))


# --- Datasets (M9) and models (M5, M6) -----------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    source: str = ""
    license: str = "unknown"
    consent: str = "unknown"


CONSENT = {"obtained", "not-required", "unknown"}


def add_dataset(
    session: Session, files: TenantStore, tenant: Tenant, name: str, content: bytes,
    provenance: Provenance, actor: Actor,
) -> Dataset:  # fmt: skip
    """Store a training corpus (encrypted) with where it came from and its consent basis."""
    from model_passport.llm.entities import load_corpus  # noqa: PLC0415 - optional extra

    if provenance.consent not in CONSENT:
        raise ServiceError(f"consent must be one of {', '.join(sorted(CONSENT))}")
    if session.scalars(
        select(Dataset).where(Dataset.tenant_id == tenant.id, Dataset.name == name)
    ).first():
        raise ServiceError(f"dataset {name!r} already exists")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "corpus.jsonl"
        path.write_bytes(content)
        try:
            records = list(load_corpus(path))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ServiceError(f"not a valid corpus (JSONL with text and entities): {exc}") from exc
    digest = hashlib.sha256(content).hexdigest()
    key = f"datasets/{digest}.jsonl"
    files.put(key, content)
    dataset = Dataset(
        tenant_id=tenant.id, name=name, source=provenance.source, license=provenance.license,
        consent=provenance.consent, sha256=digest, size_bytes=len(content), object_key=key,
        records=len(records), entities=sum(len(r.entities) for r in records),
        created_by=actor.id,
    )  # fmt: skip
    session.add(dataset)
    session.flush()
    auditlog.record(session, tenant.id, actor, "dataset.added", "dataset", dataset.id,
                    {"name": name, "sha256": digest, "records": dataset.records,
                     "license": provenance.license, "consent": provenance.consent})  # fmt: skip
    return dataset


def register_model(
    session: Session, tenant: Tenant, name: str, access: Access, base: str, description: str,
    actor: Actor,
) -> Model:  # fmt: skip
    if session.scalars(
        select(Model).where(Model.tenant_id == tenant.id, Model.name == name)
    ).first():
        raise ServiceError(f"model {name!r} already exists")
    model = Model(tenant_id=tenant.id, name=name, access=access, base=base,
                  description=description, created_by=actor.id)  # fmt: skip
    session.add(model)
    session.flush()
    auditlog.record(session, tenant.id, actor, "model.registered", "model", model.id,
                    {"name": name, "access": access.value, "base": base})  # fmt: skip
    return model


def register_version(
    session: Session, model: Model, version: str, dataset: Dataset | None, reference: Dataset,
    actor: Actor, artifact: tuple[str, str] | None = None, parent: ModelVersion | None = None,
) -> ModelVersion:  # fmt: skip
    """Record which data trained which model: the lineage behind the diligence report."""
    if any(v.version == version for v in model.versions):
        raise ServiceError(f"{model.name} already has version {version}")
    row = ModelVersion(
        tenant_id=model.tenant_id, model_id=model.id, version=version,
        dataset_id=dataset.id if dataset else None, reference_dataset_id=reference.id,
        artifact_key=artifact[0] if artifact else None,
        artifact_sha256=artifact[1] if artifact else None,
        parent_id=parent.id if parent else None, created_by=actor.id,
    )  # fmt: skip
    model.versions.append(row)  # keeps the loaded relationship in sync
    session.flush()
    auditlog.record(session, model.tenant_id, actor, "version.registered", "model_version",
                    row.id, {"model": model.name, "version": version,
                             "dataset": dataset.sha256 if dataset else None,
                             "reference": reference.sha256,
                             "parent": parent.version if parent else None})  # fmt: skip
    return row


Owned = TypeVar("Owned", Dataset, Model, ModelVersion)


def get(session: Session, kind: type[Owned], tenant_id: str, object_id: str) -> Owned:
    """An object by id, only if it belongs to ``tenant_id``."""
    found = session.get(kind, object_id)
    if found is None or found.tenant_id != tenant_id:
        raise NotFoundError(f"{kind.__name__.lower()} {object_id} not found")
    return found
