"""MP Guard inside the platform: API keys, settings, memorized values, and the request log.

**Keys** look like ``mpk_<key id>_<secret>``. Only a SHA-256 of the secret is stored. The
secrets are 256-bit random, so a fast hash is enough, and it keeps checking a key cheap on
every request. The full key is shown once, when it is created.

**The AI provider's key** is encrypted with the organization's own data key.

**Memorized values** are the fingerprints (keyed hashes) of confirmed High and Critical
findings from the organization's entity audits. The fingerprint key is derived from the
organization's data key, so fingerprints mean nothing outside the organization.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import secrets
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.orm import Session

from model_passport.core.schema import EntityAuditResult, Severity
from model_passport.guard.engine import Policy
from model_passport.platform import auditlog
from model_passport.platform.models import (
    GuardKey,
    GuardSettings,
    ModelVersion,
    Tenant,
    TenantStatus,
    now,
)
from model_passport.platform.security import decrypt, encrypt, unwrap_key

KEY_PREFIX = "mpk_"
MEMORIZED_TTL = 60.0  # seconds a tenant's memorized-value list is cached


class GuardError(ValueError):
    """A guard key, setting, or provider address is not acceptable."""


def data_key(master: bytes, tenant: Tenant) -> bytes:
    return unwrap_key(master, tenant.wrapped_key, tenant.id)


def fingerprint_key(key: bytes) -> bytes:
    """The organization's key for fingerprinting audit findings and guard traffic."""
    return hmac.new(key, b"mp-fingerprint-v1", hashlib.sha256).digest()


# --- API keys ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class IssuedKey:
    row: GuardKey
    key: str  # the only time the full key exists outside the caller's hands


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def issue_key(session: Session, tenant: Tenant, name: str, actor: auditlog.Actor) -> IssuedKey:
    name = name.strip()
    if not name:
        raise GuardError("give the key a name, such as the app that will use it")
    row = GuardKey(tenant_id=tenant.id, name=name[:200], hint="", secret_sha256="")
    row.id = str(uuid.uuid4())
    secret = secrets.token_urlsafe(32)
    key = f"{KEY_PREFIX}{uuid.UUID(row.id).hex}_{secret}"
    row.hint = key[:12]
    row.secret_sha256 = _digest(secret)
    row.created_by = actor.id
    session.add(row)
    session.flush()
    auditlog.record(
        session, tenant.id, actor, "guard.key.created", "guard_key", row.id, {"name": row.name}
    )
    return IssuedKey(row, key)


def revoke_key(session: Session, row: GuardKey, actor: auditlog.Actor) -> None:
    if row.revoked_at is None:
        row.revoked_at = now()
        auditlog.record(
            session,
            row.tenant_id,
            actor,
            "guard.key.revoked",
            "guard_key",
            row.id,
            {"name": row.name},
        )


def authenticate(session: Session, key: str) -> tuple[GuardKey, Tenant]:
    """The key and its organization; ``session`` must see all tenants."""
    if not key.startswith(KEY_PREFIX):
        raise GuardError("not an MP Guard key")
    key_hex, _, secret = key[len(KEY_PREFIX) :].partition("_")
    try:
        key_id = str(uuid.UUID(hex=key_hex))
    except ValueError as exc:
        raise GuardError("malformed MP Guard key") from exc
    row = session.get(GuardKey, key_id)
    if row is None or not hmac.compare_digest(row.secret_sha256, _digest(secret)):
        raise GuardError("invalid MP Guard key")
    if row.revoked_at is not None:
        raise GuardError("this MP Guard key was revoked")
    tenant = session.get(Tenant, row.tenant_id)
    if tenant is None or tenant.status is not TenantStatus.ACTIVE:
        raise GuardError("this organization is not active")
    return row, tenant


# --- Settings ---------------------------------------------------------------------------------


def settings_for(session: Session, tenant_id: str) -> GuardSettings:
    """The organization's settings; defaults (not yet saved) if it has none."""
    row = session.get(GuardSettings, tenant_id)
    if row is None:
        row = GuardSettings(
            tenant_id=tenant_id,
            upstream_url="https://api.openai.com/v1",
            default_model="",
            policy={},
        )
    return row


def _context(tenant_id: str) -> bytes:
    return b"guard-upstream-key:" + tenant_id.encode()


def check_upstream(url: str, allow_private: bool = False) -> str:
    """An acceptable provider address (``https://host/v1``), normalized without a trailing /."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise GuardError("the provider address must be a full URL, like https://api.openai.com/v1")
    if allow_private:
        return url.strip().rstrip("/")
    if parts.scheme != "https":
        raise GuardError("the provider address must use https")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise GuardError(f"cannot resolve {parts.hostname}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise GuardError("the provider address points to a private or local network")
    return url.strip().rstrip("/")


@dataclass(frozen=True)
class SettingsChange:
    """New settings. ``upstream_key=None`` keeps the stored key; ``""`` removes it."""

    upstream_url: str
    default_model: str = ""
    policy: dict[str, Any] | None = None
    upstream_key: str | None = None


def update_settings(
    session: Session,
    tenant: Tenant,
    key: bytes,
    actor: auditlog.Actor,
    change: SettingsChange,
    allow_private: bool = False,
) -> GuardSettings:
    try:
        parsed = Policy.from_json(change.policy)
    except (TypeError, ValueError) as exc:
        raise GuardError(f"invalid policy: {exc}") from exc
    row = settings_for(session, tenant.id)
    row.upstream_url = check_upstream(change.upstream_url, allow_private)
    row.default_model = change.default_model.strip()[:200]
    row.policy = parsed.to_json()
    if change.upstream_key is not None:
        secret = change.upstream_key.strip()
        row.upstream_key = encrypt(key, secret.encode(), _context(tenant.id)) if secret else None
    row.updated_by = actor.id
    row.updated_at = now()
    session.add(row)
    session.flush()
    details = {"upstream_url": row.upstream_url, "key_changed": change.upstream_key is not None}
    auditlog.record(
        session, tenant.id, actor, "guard.settings.updated", "guard_settings", tenant.id, details
    )
    return row


def provider_key(row: GuardSettings, key: bytes) -> str | None:
    if row.upstream_key is None:
        return None
    return decrypt(key, row.upstream_key, _context(row.tenant_id)).decode()


# --- Memorized values -------------------------------------------------------------------------


def memorized_fingerprints(session: Session, tenant_id: str) -> frozenset[str]:
    """Fingerprints of confirmed High and Critical findings across the organization's audits."""
    found: set[str] = set()
    versions = session.scalars(
        select(ModelVersion).where(
            ModelVersion.tenant_id == tenant_id, ModelVersion.audit.is_not(None)
        )
    )
    for version in versions:
        audit = EntityAuditResult.model_validate(version.audit)
        for finding in audit.findings:
            serious = finding.severity in (Severity.HIGH, Severity.CRITICAL)
            if serious and finding.confirmed is not False and finding.fingerprint:
                found.add(finding.fingerprint)
    return frozenset(found)


class MemorizedCache:
    """Per-organization fingerprint lists, refreshed every ``MEMORIZED_TTL`` seconds."""

    def __init__(self, ttl: float = MEMORIZED_TTL) -> None:
        self.ttl = ttl
        self._entries: dict[str, tuple[float, frozenset[str]]] = {}

    def get(self, session: Session, tenant_id: str) -> frozenset[str]:
        entry = self._entries.get(tenant_id)
        if entry is None or entry[0] < time.monotonic():
            entry = (time.monotonic() + self.ttl, memorized_fingerprints(session, tenant_id))
            self._entries[tenant_id] = entry
        return entry[1]
