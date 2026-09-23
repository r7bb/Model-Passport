"""FastAPI passport registry: upload, list, get, verify, append events, render.

Run with ``passport serve`` (or ``uvicorn model_passport.registry.api:app``). Configuration:

* ``PASSPORT_REGISTRY_DB``: SQLite path (default ``registry.db``)
* ``PASSPORT_REGISTRY_TOKEN``: if set, uploads and event appends need ``Authorization: Bearer``
* ``PASSPORT_TRUSTED_KEYS``: directory of trusted ``*.pub`` keys; verification reports whether
  the signing key is among them
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Annotated, Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from model_passport import __version__
from model_passport.core import identity
from model_passport.core.events import verify_events
from model_passport.core.jsonld import to_jsonld
from model_passport.core.schema import LifecycleEvent, Passport
from model_passport.core.verifier import verify_document
from model_passport.registry.db import DuplicatePassportError, RegistryDB, StoredPassport
from model_passport.report.dag import lineage_dot, pipeline_dot
from model_passport.report.render import render_html

MAX_LINEAGE_DEPTH = 20


class UploadRequest(BaseModel):
    passport: dict[str, Any]
    public_key_pem: str = Field(description="PEM Ed25519 public key that signed the passport.")


class EventRequest(BaseModel):
    event: dict[str, Any] = Field(description="A signed event from `passport monitor`.")


class VerificationResponse(BaseModel):
    passport_id: str
    ok: bool
    signature_ok: bool
    merkle_ok: bool
    fingerprint_ok: bool
    key_trusted: bool | None
    event_errors: list[str]
    artifacts_checked: bool = False
    note: str = "Artifact files are not stored in the registry; run `passport verify` locally."


def _load_key(pem: str) -> Ed25519PublicKey:
    try:
        key = load_pem_public_key(pem.encode("utf-8"))
    except ValueError as exc:
        raise HTTPException(422, f"invalid public key: {exc}") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise HTTPException(422, "public key must be Ed25519")
    return key


def _trusted_fingerprints(directory: str | None) -> set[str] | None:
    if not directory:
        return None
    fingerprints = set()
    for path in Path(directory).glob("*.pub"):
        try:
            fingerprints.add(identity.public_key_fingerprint(identity.load_public_key(path)))
        except (OSError, ValueError):
            continue
    return fingerprints


def create_app(
    db_path: str | Path | None = None,
    token: str | None = None,
    trusted_keys_dir: str | None = None,
) -> FastAPI:
    db = RegistryDB(db_path or os.environ.get("PASSPORT_REGISTRY_DB", "registry.db"))
    # An empty token (argument or environment) disables write authentication.
    token = (token if token is not None else os.environ.get("PASSPORT_REGISTRY_TOKEN")) or None
    trusted = _trusted_fingerprints(trusted_keys_dir or os.environ.get("PASSPORT_TRUSTED_KEYS"))

    app = FastAPI(title="Model Passport Registry", version=__version__)
    app.state.db = db

    def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        if token is None:
            return
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(401, "missing or invalid bearer token")

    def stored_or_404(passport_id: str) -> StoredPassport:
        stored = db.get(passport_id)
        if stored is None:
            raise HTTPException(404, f"passport {passport_id} not found")
        return stored

    def verification(stored: StoredPassport) -> VerificationResponse:
        report = verify_document(stored.document, _load_key(stored.public_key_pem))
        return VerificationResponse(
            passport_id=stored.passport_id,
            ok=report.ok,
            signature_ok=report.signature_ok,
            merkle_ok=report.merkle_ok,
            fingerprint_ok=report.fingerprint_ok,
            key_trusted=None if trusted is None else stored.key_fingerprint in trusted,
            event_errors=report.event_errors + report.errors,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/passports", status_code=201, dependencies=[Depends(require_token)])
    def upload(request: UploadRequest) -> dict[str, Any]:
        """Store a passport after checking its signature, Merkle root, key, and events."""
        report = verify_document(request.passport, _load_key(request.public_key_pem))
        if not report.ok:
            problems = report.errors + report.event_errors
            if not report.signature_ok:
                problems.append("signature is invalid")
            if not report.merkle_ok:
                problems.append("merkle root does not match artifacts")
            if not report.fingerprint_ok:
                problems.append("public key does not match the passport's fingerprint")
            raise HTTPException(
                422, {"message": "passport failed verification", "problems": problems}
            )
        try:
            stored = db.insert(request.passport, request.public_key_pem)
        except DuplicatePassportError as exc:
            raise HTTPException(409, f"passport {exc} already exists") from exc
        return stored.summary()

    @app.get("/passports")
    def list_passports(
        model_name: str | None = None,
        verdict: str | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    ) -> list[dict[str, Any]]:
        return [s.summary() for s in db.list(model_name, verdict, limit)]

    @app.get("/models")
    def list_models() -> list[dict[str, Any]]:
        return db.models()

    @app.get("/passports/{passport_id}")
    def get_passport(passport_id: str) -> dict[str, Any]:
        return stored_or_404(passport_id).document

    @app.get("/passports/{passport_id}/public-key")
    def get_public_key(passport_id: str) -> dict[str, str]:
        stored = stored_or_404(passport_id)
        return {"fingerprint": stored.key_fingerprint, "public_key_pem": stored.public_key_pem}

    @app.get("/passports/{passport_id}/verify")
    def verify(passport_id: str) -> VerificationResponse:
        return verification(stored_or_404(passport_id))

    @app.post("/passports/{passport_id}/events", dependencies=[Depends(require_token)])
    def append_event(passport_id: str, request: EventRequest) -> dict[str, Any]:
        """Append a signed event; rejected unless it extends the chain with a valid signature."""
        stored = stored_or_404(passport_id)
        LifecycleEvent.model_validate(request.event)
        document = {
            **stored.document,
            "events": [*(stored.document.get("events") or []), request.event],
        }
        errors = verify_events(document, _load_key(stored.public_key_pem))
        if errors:
            raise HTTPException(422, {"message": "event rejected", "problems": errors})
        db.replace_document(passport_id, document)
        return {"passport_id": passport_id, "events": len(document["events"])}

    @app.get("/passports/{passport_id}/lineage")
    def lineage(passport_id: str) -> dict[str, Any]:
        """Upstream models and superseded versions, resolved recursively."""
        start = stored_or_404(passport_id)
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, str]] = []
        frontier, depth = [start], 0
        while frontier and depth < MAX_LINEAGE_DEPTH:
            next_frontier = []
            for stored in frontier:
                if stored.passport_id in nodes:
                    continue
                nodes[stored.passport_id] = stored.summary()
                doc = stored.document
                parents = [(str(u), "upstream") for u in doc.get("lineage_links") or []]
                if stored.supersedes:
                    parents.append((stored.supersedes, "supersedes"))
                for parent_id, relation in parents:
                    edges.append(
                        {"from": parent_id, "to": stored.passport_id, "relation": relation}
                    )
                    parent = db.get(parent_id)
                    if parent is None:
                        nodes.setdefault(parent_id, {"passport_id": parent_id, "missing": True})
                    else:
                        next_frontier.append(parent)
            frontier, depth = next_frontier, depth + 1
        return {"root": passport_id, "nodes": list(nodes.values()), "edges": edges}

    @app.get("/passports/{passport_id}/dag")
    def dag(passport_id: str) -> dict[str, str]:
        passport = Passport.model_validate(stored_or_404(passport_id).document)
        return {"pipeline": pipeline_dot(passport), "lineage": lineage_dot(passport)}

    @app.get("/passports/{passport_id}/jsonld")
    def jsonld(passport_id: str) -> dict[str, Any]:
        return to_jsonld(Passport.model_validate(stored_or_404(passport_id).document))

    @app.get("/passports/{passport_id}/html", response_class=HTMLResponse)
    def html(passport_id: str) -> str:
        stored = stored_or_404(passport_id)
        passport = Passport.model_validate(stored.document)
        report = verify_document(stored.document, _load_key(stored.public_key_pem))
        return render_html(passport, report)

    return app


def __getattr__(name: str) -> Any:
    """``uvicorn model_passport.registry.api:app`` builds the app from the environment lazily."""
    if name == "app":
        return create_app()
    raise AttributeError(name)
