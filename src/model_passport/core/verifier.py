"""Verify a passport: artifact hashes, Merkle root, key fingerprint, and signature."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from model_passport.core import identity
from model_passport.core.schema import ArtifactKind, Passport


class ArtifactStatus(StrEnum):
    OK = "ok"
    MODIFIED = "modified"
    MISSING = "missing"


@dataclass
class ArtifactCheck:
    path: str
    kind: ArtifactKind
    expected_sha256: str
    actual_sha256: str | None
    status: ArtifactStatus


@dataclass
class VerificationReport:
    artifacts: list[ArtifactCheck] = field(default_factory=list)
    merkle_ok: bool = False
    fingerprint_ok: bool = False
    signature_ok: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def changed(self) -> list[ArtifactCheck]:
        return [a for a in self.artifacts if a.status is not ArtifactStatus.OK]

    @property
    def ok(self) -> bool:
        return (
            not self.errors
            and not self.changed
            and self.merkle_ok
            and self.fingerprint_ok
            and self.signature_ok
        )


def _check_artifact(root: Path, path: str, kind: ArtifactKind, expected: str) -> ArtifactCheck:
    full = root / path
    if not full.is_file():
        return ArtifactCheck(path, kind, expected, None, ArtifactStatus.MISSING)
    actual = identity.sha256_file(full)
    status = ArtifactStatus.OK if actual == expected else ArtifactStatus.MODIFIED
    return ArtifactCheck(path, kind, expected, actual, status)


def verify_passport(passport_path: Path, public_key_path: Path, root: Path) -> VerificationReport:
    """Verify ``passport_path`` against a trusted public key and the files under ``root``.

    The signature is checked over the raw JSON as stored (not a re-serialized model), so any
    edit to the file, including unknown fields, invalidates it.
    """
    report = VerificationReport()
    try:
        raw: dict[str, Any] = json.loads(passport_path.read_text(encoding="utf-8"))
        passport = Passport.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        report.errors.append(f"cannot read passport: {exc}")
        return report

    try:
        public_key = identity.load_public_key(public_key_path)
    except (OSError, ValueError) as exc:
        report.errors.append(f"cannot load public key: {exc}")
        return report

    report.artifacts = [_check_artifact(root, a.path, a.kind, a.sha256) for a in passport.artifacts]
    report.merkle_ok = (
        identity.merkle_root(a.sha256 for a in passport.artifacts) == passport.identity.merkle_root
    )
    report.fingerprint_ok = (
        identity.public_key_fingerprint(public_key) == passport.identity.public_key_fingerprint
    )
    signature = passport.identity.signature
    report.signature_ok = bool(signature) and identity.verify_signature(
        public_key, identity.signing_payload(raw), signature or ""
    )
    return report
