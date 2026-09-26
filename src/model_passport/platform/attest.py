"""Signed attestations for model versions: the verifiable core of the diligence report.

Each tenant has its own Ed25519 signing key. The private key is kept only in the tenant's
encrypted storage; the public key is published on the tenant record. An attestation records
what a version is and how it was checked, and is signed over its canonical JSON:

- the model and version, and the hash of its artifact;
- the training and reference data, with source, license, and consent;
- the parent version, by attestation hash, so versions form a verifiable chain;
- the audit summary and the gate verdict;
- the current head of the tenant's audit log, anchoring the attestation in that history.
"""

from __future__ import annotations

from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from sqlalchemy.orm import Session

from model_passport.core import identity
from model_passport.core.schema import PolicyResult
from model_passport.platform import auditlog
from model_passport.platform.models import Dataset, ModelVersion, Tenant, now
from model_passport.platform.storage import StorageError, TenantStore

KEY_OBJECT = "keys/signing-key.pem"
KIND = "mp.version-attestation/v1"


class AttestationError(ValueError):
    """The attestation is invalid or cannot be produced."""


def signing_key(tenant: Tenant, files: TenantStore) -> Ed25519PrivateKey:
    """The tenant's signing key, created (and its public half published) on first use."""
    try:
        pem = files.get(KEY_OBJECT)
    except StorageError:
        key = Ed25519PrivateKey.generate()
        files.put(
            KEY_OBJECT,
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
        )
        tenant.public_key = (
            key.public_key()
            .public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            )
            .decode()
        )
        return key
    loaded = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(loaded, Ed25519PrivateKey):
        raise AttestationError("tenant signing key is not Ed25519")
    return loaded


def _dataset(dataset: Dataset | None) -> dict[str, Any] | None:
    if dataset is None:
        return None
    return {
        "name": dataset.name,
        "sha256": dataset.sha256,
        "records": dataset.records,
        "source": dataset.source,
        "license": dataset.license,
        "consent": dataset.consent,
    }


def _load(session: Session, dataset_id: str | None) -> Dataset | None:
    return session.get(Dataset, dataset_id) if dataset_id else None


def attest(
    session: Session,
    tenant: Tenant,
    files: TenantStore,
    version: ModelVersion,
    gate: PolicyResult | None,
) -> dict[str, Any]:
    """Sign an attestation for ``version``, store it on the row, and log it."""
    key = signing_key(tenant, files)
    parent = session.get(ModelVersion, version.parent_id) if version.parent_id else None
    head = auditlog.chain(session, tenant.id)[-1:] or []
    audit = version.audit or {}
    document: dict[str, Any] = {
        "type": KIND,
        "tenant": tenant.slug,
        "model": version.model.name,
        "access": version.model.access.value,
        "base": version.model.base,
        "version": version.version,
        "version_id": version.id,
        "state": version.state.value,
        "issued_at": now().isoformat(),
        "artifact_sha256": version.artifact_sha256,
        "training_data": _dataset(_load(session, version.dataset_id)),
        "reference_data": _dataset(_load(session, version.reference_dataset_id)),
        "parent": None
        if parent is None
        else {
            "version": parent.version,
            "attestation_sha256": parent.attestation_sha256,
        },
        "audit": {
            k: audit.get(k)
            for k in (
                "model",
                "access",
                "primary_method",
                "entities_audited",
                "auc",
                "tpr_at_fpr",
                "severity_counts",
                "taxonomy",
            )
        },
        "gate": None if gate is None else gate.model_dump(mode="json"),
        "audit_log_head": None if not head else {"seq": head[0].seq, "hash": head[0].hash},
        "public_key_fingerprint": identity.public_key_fingerprint(key.public_key()),
    }
    document["signature"] = identity.sign(key, identity.canonical_json(document))
    version.attestation = document
    version.attestation_sha256 = identity.sha256_bytes(identity.canonical_json(document))
    auditlog.record(
        session,
        tenant.id,
        auditlog.SYSTEM,
        "version.attested",
        "model_version",
        version.id,
        {"sha256": version.attestation_sha256},
    )
    return document


def verify(document: dict[str, Any], public_key_pem: str) -> bool:
    """True if ``document`` is unchanged and signed by the tenant key ``public_key_pem``."""
    key = serialization.load_pem_public_key(public_key_pem.encode())
    if not isinstance(key, Ed25519PublicKey):
        return False
    body = {k: v for k, v in document.items() if k != "signature"}
    signature = document.get("signature")
    return isinstance(signature, str) and identity.verify_signature(
        key, identity.canonical_json(body), signature
    )
