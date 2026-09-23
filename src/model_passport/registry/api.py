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
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from model_passport import __version__
from model_passport.core import identity
from model_passport.core.events import verify_events
from model_passport.core.jsonld import to_jsonld
from model_passport.core.schema import LifecycleEvent, Passport
from model_passport.core.verifier import VerificationReport, verify_document
from model_passport.registry.db import DuplicatePassportError, RegistryDB, StoredPassport
from model_passport.report.dag import lineage_dot, pipeline_dot
from model_passport.report.render import render_html

MAX_LINEAGE_DEPTH = 20


# --- Models --------------------------------------------------------------------------------


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


@dataclass(frozen=True)
class RegistryContext:
    db: RegistryDB
    token: str | None
    trusted_fingerprints: set[str] | None


# --- Helpers and dependencies --------------------------------------------------------------


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


def _problems(report: VerificationReport) -> list[str]:
    problems = report.errors + report.event_errors
    if not report.signature_ok:
        problems.append("signature is invalid")
    if not report.merkle_ok:
        problems.append("merkle root does not match artifacts")
    if not report.fingerprint_ok:
        problems.append("public key does not match the passport's fingerprint")
    return problems


def get_context(request: Request) -> RegistryContext:
    context: RegistryContext = request.app.state.registry
    return context


Context = Annotated[RegistryContext, Depends(get_context)]


def require_token(context: Context, authorization: Annotated[str | None, Header()] = None) -> None:
    if context.token is None:
        return
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied.encode(), context.token.encode()):
        raise HTTPException(401, "missing or invalid bearer token")


def get_stored(passport_id: str, context: Context) -> StoredPassport:
    stored = context.db.get(passport_id)
    if stored is None:
        raise HTTPException(404, f"passport {passport_id} not found")
    return stored


Stored = Annotated[StoredPassport, Depends(get_stored)]
router = APIRouter()


# --- Routes --------------------------------------------------------------------------------


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.post("/passports", status_code=201, dependencies=[Depends(require_token)])
def upload(request: UploadRequest, context: Context) -> dict[str, Any]:
    """Store a passport after checking its signature, Merkle root, key, and events."""
    report = verify_document(request.passport, _load_key(request.public_key_pem))
    if not report.ok:
        detail = {"message": "passport failed verification", "problems": _problems(report)}
        raise HTTPException(422, detail)
    try:
        stored = context.db.insert(request.passport, request.public_key_pem)
    except DuplicatePassportError as exc:
        raise HTTPException(409, f"passport {exc} already exists") from exc
    return stored.summary()


@router.get("/passports")
def list_passports(
    context: Context,
    model_name: str | None = None,
    verdict: str | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> list[dict[str, Any]]:
    return [s.summary() for s in context.db.search(model_name, verdict, limit)]


@router.get("/models")
def list_models(context: Context) -> list[dict[str, Any]]:
    return context.db.models()


@router.get("/passports/{passport_id}")
def get_passport(stored: Stored) -> dict[str, Any]:
    return stored.document


@router.get("/passports/{passport_id}/public-key")
def get_public_key(stored: Stored) -> dict[str, str]:
    return {"fingerprint": stored.key_fingerprint, "public_key_pem": stored.public_key_pem}


@router.get("/passports/{passport_id}/verify")
def verify(stored: Stored, context: Context) -> VerificationResponse:
    report = verify_document(stored.document, _load_key(stored.public_key_pem))
    trusted = context.trusted_fingerprints
    return VerificationResponse(
        passport_id=stored.passport_id,
        ok=report.ok,
        signature_ok=report.signature_ok,
        merkle_ok=report.merkle_ok,
        fingerprint_ok=report.fingerprint_ok,
        key_trusted=None if trusted is None else stored.key_fingerprint in trusted,
        event_errors=report.event_errors + report.errors,
    )


@router.post("/passports/{passport_id}/events", dependencies=[Depends(require_token)])
def append_event(stored: Stored, request: EventRequest, context: Context) -> dict[str, Any]:
    """Append a signed event; rejected unless it extends the chain with a valid signature."""
    LifecycleEvent.model_validate(request.event)
    events = [*(stored.document.get("events") or []), request.event]
    document = {**stored.document, "events": events}
    errors = verify_events(document, _load_key(stored.public_key_pem))
    if errors:
        raise HTTPException(422, {"message": "event rejected", "problems": errors})
    context.db.replace_document(stored.passport_id, document)
    return {"passport_id": stored.passport_id, "events": len(events)}


@router.get("/passports/{passport_id}/lineage")
def lineage(stored: Stored, context: Context) -> dict[str, Any]:
    """Upstream models and superseded versions, resolved breadth-first."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    frontier = [stored]
    for _ in range(MAX_LINEAGE_DEPTH):
        frontier = [p for p in frontier if p.passport_id not in nodes]
        if not frontier:
            break
        next_frontier: list[StoredPassport] = []
        for current in frontier:
            nodes[current.passport_id] = current.summary()
            for parent_id, relation in _parents(current):
                edges.append({"from": parent_id, "to": current.passport_id, "relation": relation})
                parent = context.db.get(parent_id)
                if parent is None:
                    nodes.setdefault(parent_id, {"passport_id": parent_id, "missing": True})
                else:
                    next_frontier.append(parent)
        frontier = next_frontier
    return {"root": stored.passport_id, "nodes": list(nodes.values()), "edges": edges}


def _parents(stored: StoredPassport) -> list[tuple[str, str]]:
    parents = [(str(u), "upstream") for u in stored.document.get("lineage_links") or []]
    if stored.supersedes:
        parents.append((stored.supersedes, "supersedes"))
    return parents


@router.get("/passports/{passport_id}/dag")
def dag(stored: Stored) -> dict[str, str]:
    passport = Passport.model_validate(stored.document)
    return {"pipeline": pipeline_dot(passport), "lineage": lineage_dot(passport)}


@router.get("/passports/{passport_id}/jsonld")
def jsonld(stored: Stored) -> dict[str, Any]:
    return to_jsonld(Passport.model_validate(stored.document))


@router.get("/passports/{passport_id}/html", response_class=HTMLResponse)
def html(stored: Stored) -> str:
    passport = Passport.model_validate(stored.document)
    report = verify_document(stored.document, _load_key(stored.public_key_pem))
    return render_html(passport, report)


# --- App factory ---------------------------------------------------------------------------


def create_app(
    db_path: str | Path | None = None,
    token: str | None = None,
    trusted_keys_dir: str | None = None,
) -> FastAPI:
    # An empty token (argument or environment) disables write authentication.
    resolved_token = (
        token if token is not None else os.environ.get("PASSPORT_REGISTRY_TOKEN")
    ) or None
    app = FastAPI(title="Model Passport Registry", version=__version__)
    app.state.registry = RegistryContext(
        db=RegistryDB(db_path or os.environ.get("PASSPORT_REGISTRY_DB", "registry.db")),
        token=resolved_token,
        trusted_fingerprints=_trusted_fingerprints(
            trusted_keys_dir or os.environ.get("PASSPORT_TRUSTED_KEYS")
        ),
    )
    app.include_router(router)
    return app


def __getattr__(name: str) -> Any:
    """``uvicorn model_passport.registry.api:app`` builds the app from the environment lazily."""
    if name == "app":
        return create_app()
    raise AttributeError(name)
