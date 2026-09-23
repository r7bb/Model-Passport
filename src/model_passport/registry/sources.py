"""Where the dashboard reads passports from: the registry API or local files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from model_passport.core import identity
from model_passport.core.schema import Passport
from model_passport.core.verifier import VerificationReport, verify_document
from model_passport.registry.client import RegistryClient


@dataclass(frozen=True)
class Entry:
    passport_id: str
    model_name: str
    version: str
    created_at: str
    verdict: str | None
    sequence: int | None = None

    @property
    def label(self) -> str:
        revision = f" · rev {self.sequence}" if self.sequence else ""
        verdict = (self.verdict or "no policy").upper()
        return f"v{self.version}{revision} · {verdict} · {self.created_at[:16].replace('T', ' ')}"


class PassportSource(Protocol):
    def entries(self) -> list[Entry]: ...

    def document(self, passport_id: str) -> dict[str, Any]: ...

    def verification(self, passport_id: str) -> dict[str, Any]: ...


def _entry(doc: dict[str, Any]) -> Entry:
    ident = doc["identity"]
    return Entry(
        passport_id=ident["passport_id"],
        model_name=ident["model_name"],
        version=ident["version"],
        created_at=ident["created_at"],
        verdict=(doc.get("policy") or {}).get("verdict"),
        sequence=(doc.get("revision") or {}).get("sequence"),
    )


class LocalSource:
    """Passports in a directory (``*.json``) plus ``.passport/history``."""

    def __init__(self, root: Path, public_key: Path | None = None) -> None:
        self.root = root
        self.public_key = public_key or root / ".passport/signing_key.pub"
        self._docs: dict[str, dict[str, Any]] = {}
        for path in [*root.glob("*.json"), *(root / ".passport/history").glob("*.json")]:
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                Passport.model_validate(doc)
            except (OSError, json.JSONDecodeError, ValidationError, TypeError):
                continue
            # The working copy (with the newest events) wins over its archived copy.
            self._docs.setdefault(doc["identity"]["passport_id"], doc)

    def entries(self) -> list[Entry]:
        return sorted((_entry(d) for d in self._docs.values()), key=lambda e: e.created_at,
                      reverse=True)  # fmt: skip

    def document(self, passport_id: str) -> dict[str, Any]:
        return self._docs[passport_id]

    def verification(self, passport_id: str) -> dict[str, Any]:
        try:
            key = identity.load_public_key(self.public_key)
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": f"no public key: {exc}"}
        return _report_dict(verify_document(self._docs[passport_id], key))


class RegistrySource:
    def __init__(self, client: RegistryClient) -> None:
        self.client = client

    def entries(self) -> list[Entry]:
        return [
            Entry(
                passport_id=s["passport_id"],
                model_name=s["model_name"],
                version=s["version"],
                created_at=s["created_at"],
                verdict=s.get("verdict"),
                sequence=None,
            )
            for s in self.client.search()
        ]

    def document(self, passport_id: str) -> dict[str, Any]:
        return self.client.get(passport_id)

    def verification(self, passport_id: str) -> dict[str, Any]:
        return self.client.verify(passport_id)


def _report_dict(report: VerificationReport) -> dict[str, Any]:
    return {
        "ok": report.ok,
        "signature_ok": report.signature_ok,
        "merkle_ok": report.merkle_ok,
        "fingerprint_ok": report.fingerprint_ok,
        "event_errors": report.event_errors + report.errors,
    }
