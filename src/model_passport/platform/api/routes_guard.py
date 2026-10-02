"""MP Guard over HTTP: the drop-in endpoint for apps, and its settings for the dashboard.

Apps keep their OpenAI SDK and change only two settings::

    client = OpenAI(base_url="https://<organization>.<domain>/guard/v1", api_key="mpk_...")

Requests are protected (personal values masked), forwarded to the organization's AI provider
with the provider key MP stores for it, and the reply is reviewed before the app sees it.
Errors use OpenAI's error format, so SDKs surface them as usual.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Annotated, Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from model_passport.guard.chat import protect_messages, review_completion
from model_passport.guard.engine import (
    BlockedError,
    Guard,
    PatternDetector,
    Policy,
    Report,
    Vault,
    fingerprint,
)
from model_passport.platform import guard as service
from model_passport.platform.api.deps import AppState, Ctx, State, require
from model_passport.platform.api.schemas import (
    GuardEventOut,
    GuardKeyCreated,
    GuardKeyIn,
    GuardKeyOut,
    GuardPlaygroundIn,
    GuardSettingsIn,
    GuardSettingsOut,
)
from model_passport.platform.db import ALL_TENANTS, scoped_session
from model_passport.platform.models import GuardEvent, GuardKey, GuardSettings, Tenant, now
from model_passport.platform.rbac import Permission
from model_passport.platform.security import RateLimiter

P = Permission
proxy = APIRouter(prefix="/guard/v1", tags=["MP Guard endpoint"])
router = APIRouter(prefix="/guard", tags=["MP Guard"])


def ctx(permission: P) -> Any:
    return Depends(require(permission))


def _error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(_body(message, code), status_code=status)


# --- The pipeline -----------------------------------------------------------------------------


@dataclass
class Run:
    status: int
    body: dict[str, Any]
    protected: list[dict[str, Any]] | None = None
    report: Report = field(default_factory=Report)


def _engine(app: AppState, session: Session, tenant: Tenant, policy: Policy) -> Guard:
    key = service.fingerprint_key(service.data_key(app.settings.master_key, tenant))
    known = app.memorized.get(session, tenant.id)
    return Guard(
        policy, PatternDetector(), lambda kind, value: fingerprint(key, kind, value) in known
    )


def _outcome(report: Report) -> str:
    acted = report.masked or report.redacted or report.memorized
    return "protected" if acted else "passed"


class RunStoppedError(Exception):
    """End a run early with this response (``upstream`` is the provider's status, if any)."""

    def __init__(
        self,
        status: int,
        message: str,
        code: str,
        outcome: str = "error",
        upstream: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status, self.code, self.outcome, self.upstream = status, code, outcome, upstream
        self.payload: dict[str, Any] = _body(message, code)


def _messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RunStoppedError(400, "messages must be a non-empty list", "invalid_request")
    if body.get("stream"):
        raise RunStoppedError(
            400,
            "streaming replies are not supported yet; set stream to false",
            "stream_unsupported",
        )
    return messages


def _check_limits(body: dict[str, Any], max_tokens: int, max_n: int) -> dict[str, Any]:
    """Refuse requests that could run up cost; return the extra fields to send upstream.

    Over-limit values are rejected (422), not clamped: silently changing a client's request
    would hide the problem, and a gateway should say what it will not do. A request that names
    no token limit gets ``max_tokens`` so the provider's own default cannot run unbounded.
    """
    for name in ("max_tokens", "max_completion_tokens"):
        value = body.get(name)
        if value is not None and (not _is_count(value) or value > max_tokens):
            message = f"{name} must be a whole number from 1 to {max_tokens}"
            raise RunStoppedError(422, message, "invalid_request")
    n = body.get("n")
    if n is not None and (not _is_count(n) or n > max_n):
        raise RunStoppedError(422, f"n must be a whole number from 1 to {max_n}", "invalid_request")
    named = body.get("max_tokens") is not None or body.get("max_completion_tokens") is not None
    return {} if named else {"max_tokens": max_tokens}


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _pinned_targets(url: str, allow_private: bool) -> list[tuple[str, dict[str, Any]]]:
    """Where to send the request: ``(url, request options)`` per address that passed the check.

    The name is resolved and every address checked now, at request time, then the connection is
    pinned to the checked address so the HTTP client cannot resolve the name again (DNS
    rebinding). The ``Host`` header and TLS server name stay the real ones, and TLS
    verification stays on. When private networks are allowed there is nothing to check or pin.
    """
    if allow_private:
        return [(url, {})]
    parts = urlsplit(url)
    host = parts.hostname or ""
    blocked = RunStoppedError(502, "the provider address is not allowed", "provider_blocked")
    if parts.scheme != "https":
        raise blocked
    try:
        addresses = service.public_addresses(host, parts.port or 443)
    except service.GuardError as exc:
        raise blocked from exc
    shown = f"[{host}]" if ":" in host else host
    netloc = f"{shown}:{parts.port}" if parts.port else shown
    targets = []
    for address in addresses:
        literal = f"[{address}]" if ":" in address else address
        pinned = parts._replace(netloc=f"{literal}:{parts.port}" if parts.port else literal)
        options = {"headers": {"Host": netloc}, "extensions": {"sni_hostname": host}}
        targets.append((urlunsplit(pinned), options))
    return targets


def _forward(
    app: AppState, url: str, key: str, payload: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    targets = _pinned_targets(url, app.settings.guard_private_upstreams)
    response = None
    for target, options in targets:
        try:
            response = app.guard_http.post(
                f"{target}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {key}", **options.get("headers", {})},
                extensions=options.get("extensions"),
            )
            break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            failure: httpx.HTTPError = exc  # try the next checked address
        except httpx.HTTPError as exc:
            raise _unreachable(exc) from exc
    if response is None:
        raise _unreachable(failure)
    try:
        completion = response.json()
    except ValueError as exc:
        message = "the provider sent a reply that is not JSON"
        raise RunStoppedError(502, message, "bad_reply", upstream=response.status_code) from exc
    if response.status_code >= 400 or not isinstance(completion, dict):
        stop = RunStoppedError(
            response.status_code, "provider error", "provider_error", upstream=response.status_code
        )
        if isinstance(completion, dict):
            stop.payload = completion  # pass the provider's own error through
        raise stop
    return response.status_code, completion


def _unreachable(exc: httpx.HTTPError) -> RunStoppedError:
    return RunStoppedError(
        502, f"provider unreachable ({type(exc).__name__})", "provider_unreachable"
    )


def _protect(
    engine: Guard, messages: list[dict[str, Any]], vault: Vault, report: Report
) -> list[dict[str, Any]]:
    try:
        return protect_messages(engine, messages, vault, report)
    except BlockedError as exc:
        raise RunStoppedError(400, str(exc), "pii_blocked", "blocked") from exc


def _provider_key(settings: GuardSettings, tenant_key: bytes, model: str) -> str:
    key = service.provider_key(settings, tenant_key)
    if not key:
        message = "add your AI provider's API key in MP Guard settings"
        raise RunStoppedError(503, message, "provider_not_configured")
    if not model:
        raise RunStoppedError(400, "name a model, or set a default model", "no_model")
    return key


def _review(
    engine: Guard, completion: dict[str, Any], vault: Vault, report: Report, upstream: int
) -> dict[str, Any]:
    try:
        return review_completion(engine, completion, vault, report)
    except BlockedError as exc:
        raise RunStoppedError(422, str(exc), "pii_blocked_reply", "blocked", upstream) from exc


def run_chat(
    app: AppState,
    session: Session,
    tenant: Tenant,
    body: dict[str, Any],
    *,
    source: str,
    key_id: str | None = None,
) -> Run:
    """Protect, forward, and review one chat completion; every run is logged (counts only)."""
    start = time.monotonic()
    settings = service.settings_for(session, tenant.id)
    tenant_key = service.data_key(app.settings.master_key, tenant)
    engine = _engine(app, session, tenant, Policy.from_json(settings.policy))
    model = str(body.get("model") or settings.default_model)
    run, vault, upstream = Run(200, {}), Vault(), None
    outcome, detail = "passed", ""
    try:
        messages = _messages(body)
        run.protected = _protect(engine, messages, vault, run.report)
        extra = _check_limits(body, app.settings.guard_max_tokens, app.settings.guard_max_n)
        key = _provider_key(settings, tenant_key, model)
        payload = {**body, **extra, "model": model, "messages": run.protected}
        upstream, completion = _forward(app, settings.upstream_url, key, payload)
        run.body = _review(engine, completion, vault, run.report, upstream)
        outcome = _outcome(run.report)
    except RunStoppedError as stop:
        run.status, run.body, outcome = stop.status, stop.payload, stop.outcome
        upstream, detail = stop.upstream or upstream, str(stop)
    session.add(
        GuardEvent(
            tenant_id=tenant.id,
            key_id=key_id,
            source=source,
            model=model[:200],
            outcome=outcome,
            upstream_status=upstream,
            latency_ms=int((time.monotonic() - start) * 1000),
            report=run.report.to_json(),
            detail=detail[:500],
        )
    )
    return run


def _body(message: str, code: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": "mp_guard", "code": code}}


# --- The endpoint apps call -------------------------------------------------------------------


def _caller(app: AppState, authorization: str | None) -> tuple[str, Tenant]:
    token = (authorization or "").removeprefix("Bearer ").strip()
    with scoped_session(app.factory, ALL_TENANTS) as session:
        try:
            row, tenant = service.authenticate(session, token)
        except service.GuardError as exc:
            raise HTTPException(401, str(exc)) from exc
        row.last_used_at = now()
        session.expunge(tenant)
        return row.id, tenant


Authorization = Annotated[str | None, Header()]

# Request rates are counted in memory, per process (see RateLimiter). The limiters are made on
# first use from the app's settings and kept on ``app.state``.
_limiter_lock = threading.Lock()


@dataclass
class Admitted:
    key_id: str
    tenant: Tenant
    body: dict[str, Any]


def _limiters(request: Request, app: AppState) -> tuple[RateLimiter, RateLimiter]:
    with _limiter_lock:
        made = getattr(request.app.state, "guard_limiters", None)
        if made is None:
            settings = app.settings
            made = (
                RateLimiter(settings.guard_rate_per_key),
                RateLimiter(settings.guard_rate_per_ip),
            )
            request.app.state.guard_limiters = made
    return made


def _too_many(wait: int) -> JSONResponse:
    response = _error(429, "too many requests; slow down", "rate_limited")
    response.headers["Retry-After"] = str(wait)
    return response


async def _read_json(request: Request, limit: int) -> dict[str, Any] | JSONResponse:
    """Read the body, never holding more than ``limit`` bytes, and parse it as a JSON object."""
    too_large = _error(413, f"request body is over {limit} bytes", "request_too_large")
    declared = request.headers.get("content-length", "")
    if declared.isascii() and declared.isdigit() and int(declared) > limit:
        return too_large
    received = bytearray()
    async for chunk in request.stream():  # Content-Length may be absent or wrong
        received += chunk
        if len(received) > limit:
            return too_large
    try:
        body = json.loads(received)
    except (ValueError, RecursionError):  # malformed, or nested too deeply to parse
        body = None
    if not isinstance(body, dict):
        return _error(400, "send a JSON object", "invalid_request")
    return body


async def _admit(
    request: Request, app: AppState, authorization: str | None
) -> Admitted | JSONResponse:
    """Rate-limit by address, authenticate, rate-limit by key, then read the body.

    Only ``request.client.host`` identifies the address: the app has no trusted-proxy setting,
    so ``X-Forwarded-For`` is ignored (anyone could set it to dodge the limit).
    """
    per_key, per_ip = _limiters(request, app)
    wait = per_ip.check(request.client.host if request.client else "unknown")
    if wait:
        return _too_many(wait)
    try:
        key_id, tenant = await run_in_threadpool(_caller, app, authorization)
    except HTTPException as exc:
        return _error(401, str(exc.detail), "invalid_api_key")
    wait = per_key.check(key_id)
    if wait:
        return _too_many(wait)
    body = await _read_json(request, app.settings.guard_max_body_bytes)
    if isinstance(body, JSONResponse):
        return body
    return Admitted(key_id, tenant, body)


def _chat(app: AppState, admitted: Admitted) -> JSONResponse:
    with scoped_session(app.factory, admitted.tenant.id) as session:
        run = run_chat(
            app, session, admitted.tenant, admitted.body, source="api", key_id=admitted.key_id
        )
    headers = {
        "X-MP-Guard": ", ".join(f"{kind}={sum(c.values())}" for kind, c in _totals(run.report))
    }
    return JSONResponse(run.body, status_code=run.status, headers=headers)


@proxy.post("/chat/completions", response_model=None)
async def chat_completions(
    request: Request, app: State, authorization: Authorization = None
) -> JSONResponse:
    """OpenAI-compatible chat completions, with personal data kept from the model."""
    admitted = await _admit(request, app, authorization)
    if isinstance(admitted, JSONResponse):
        return admitted
    return await run_in_threadpool(_chat, app, admitted)


def _totals(report: Report) -> list[tuple[str, Counter[str]]]:
    return [
        ("masked", report.masked),
        ("redacted", report.redacted),
        ("memorized", report.memorized),
    ]


@proxy.post("/scan", response_model=None)
async def scan(request: Request, app: State, authorization: Authorization = None) -> JSONResponse:
    """Check text without calling a model, e.g. a tool's result before an agent uses it.

    ``direction`` is ``request`` (text going to a model: values are replaced) or ``reply``
    (text coming from one: leaks are handled). The reply is the text to use instead.
    """
    admitted = await _admit(request, app, authorization)
    if isinstance(admitted, JSONResponse):
        return admitted
    return await run_in_threadpool(_scan, app, admitted)


def _scan(app: AppState, admitted: Admitted) -> JSONResponse:
    body, tenant, key_id = admitted.body, admitted.tenant, admitted.key_id
    text, direction = body.get("text"), body.get("direction", "request")
    if not isinstance(text, str) or direction not in ("request", "reply"):
        return _error(400, "send text, and direction 'request' or 'reply'", "invalid_request")
    with scoped_session(app.factory, tenant.id) as session:
        settings = service.settings_for(session, tenant.id)
        engine = _engine(app, session, tenant, Policy.from_json(settings.policy))
        report, start = Report(), time.monotonic()
        status, payload, outcome = 200, {}, "passed"
        try:
            if direction == "request":
                result = engine.protect(text, Vault(), report)
            else:
                result = engine.review(text, Vault(), report)
            payload = {"text": result, "blocked": False, "report": report.to_json()}
            outcome = _outcome(report)
        except BlockedError as exc:
            status, outcome = 200, "blocked"
            payload = {
                "text": None,
                "blocked": True,
                "types": exc.types,
                "report": report.to_json(),
            }
        session.add(
            GuardEvent(
                tenant_id=tenant.id,
                key_id=key_id,
                source="scan",
                outcome=outcome,
                latency_ms=int((time.monotonic() - start) * 1000),
                report=report.to_json(),
            )
        )
    return JSONResponse(payload, status_code=status)


# --- Settings, keys, and activity for the dashboard -------------------------------------------


@router.get("/settings", response_model=GuardSettingsOut)
def get_settings(c: Annotated[Ctx, ctx(P.GUARD_READ)]) -> GuardSettingsOut:
    row = service.settings_for(c.session, c.tenant.id)
    policy = Policy.from_json(row.policy)
    return GuardSettingsOut(
        upstream_url=row.upstream_url,
        has_upstream_key=row.upstream_key is not None,
        default_model=row.default_model,
        policy=policy.to_json(),
        entity_types=list(PatternDetector.types),
        effective={
            kind: {"request": policy.for_request(kind).value, "reply": policy.for_reply(kind).value}
            for kind in PatternDetector.types
        },
        endpoint="/guard/v1",
    )


@router.put("/settings", response_model=GuardSettingsOut)
def put_settings(
    body: GuardSettingsIn, c: Annotated[Ctx, ctx(P.GUARD_MANAGE)], app: State
) -> GuardSettingsOut:
    change = service.SettingsChange(
        upstream_url=body.upstream_url,
        default_model=body.default_model,
        policy=body.policy,
        upstream_key=body.upstream_key,
    )
    try:
        service.update_settings(
            c.session,
            c.tenant,
            service.data_key(app.settings.master_key, c.tenant),
            c.actor,
            change,
            app.settings.guard_private_upstreams,
        )
    except service.GuardError as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_settings(c)


@router.get("/keys", response_model=list[GuardKeyOut])
def list_keys(c: Annotated[Ctx, ctx(P.GUARD_READ)]) -> list[GuardKey]:
    return list(
        c.session.scalars(
            select(GuardKey)
            .where(GuardKey.tenant_id == c.tenant.id)
            .order_by(GuardKey.created_at.desc())
        )
    )


@router.post("/keys", response_model=GuardKeyCreated, status_code=201)
def create_key(body: GuardKeyIn, c: Annotated[Ctx, ctx(P.GUARD_MANAGE)]) -> GuardKeyCreated:
    try:
        issued = service.issue_key(c.session, c.tenant, body.name, c.actor)
    except service.GuardError as exc:
        raise HTTPException(400, str(exc)) from exc
    return GuardKeyCreated(**GuardKeyOut.model_validate(issued.row).model_dump(), key=issued.key)


@router.delete("/keys/{key_id}", status_code=204)
def revoke(key_id: str, c: Annotated[Ctx, ctx(P.GUARD_MANAGE)]) -> None:
    row = c.session.get(GuardKey, key_id)
    if row is None or row.tenant_id != c.tenant.id:
        raise HTTPException(404, "key not found")
    service.revoke_key(c.session, row, c.actor)


@router.get("/events", response_model=list[GuardEventOut])
def events(c: Annotated[Ctx, ctx(P.GUARD_READ)], limit: int = 100) -> list[GuardEvent]:
    return list(
        c.session.scalars(
            select(GuardEvent)
            .where(GuardEvent.tenant_id == c.tenant.id)
            .order_by(GuardEvent.created_at.desc())
            .limit(min(max(limit, 1), 1000))
        )
    )


@router.get("/stats")
def stats(c: Annotated[Ctx, ctx(P.GUARD_READ)], days: int = 7) -> dict[str, Any]:
    """Totals and a per-day series for the last ``days`` days (counts only)."""
    since = now() - timedelta(days=min(max(days, 1), 90))
    rows = c.session.scalars(
        select(GuardEvent).where(
            GuardEvent.tenant_id == c.tenant.id, GuardEvent.created_at >= since
        )
    ).all()
    outcomes: Counter[str] = Counter(r.outcome for r in rows)
    sums: dict[str, Counter[str]] = {
        k: Counter() for k in ("masked", "redacted", "leaked", "memorized")
    }
    per_day: dict[str, Counter[str]] = {}
    latencies = sorted(r.latency_ms for r in rows if r.source != "playground")
    for row in rows:
        for kind, counts in (row.report or {}).items():
            if kind in sums:
                sums[kind].update(counts)
        day = row.created_at.date().isoformat()
        per_day.setdefault(day, Counter())["requests"] += 1
        per_day[day][row.outcome] += 1
    return {
        "days": days,
        "requests": len(rows),
        "outcomes": dict(outcomes),
        "values": {kind: sum(c.values()) for kind, c in sums.items()},
        "by_type": {kind: dict(c.most_common()) for kind, c in sums.items()},
        "per_day": [{"day": day, **counts} for day, counts in sorted(per_day.items())],
        "median_latency_ms": latencies[len(latencies) // 2] if latencies else None,
    }


@router.post("/playground")
def playground(
    body: GuardPlaygroundIn, c: Annotated[Ctx, ctx(P.GUARD_MANAGE)], app: State
) -> dict[str, Any]:
    """What the model would see for this text, and (with ``send``) the reviewed reply."""
    settings = service.settings_for(c.session, c.tenant.id)
    policy = Policy.from_json(settings.policy)
    if not body.send:
        engine = _engine(app, c.session, c.tenant, policy)
        report = Report()
        try:
            seen = engine.protect(body.text, Vault(), report)
        except BlockedError as exc:
            return {"blocked": True, "message": str(exc), "report": report.to_json()}
        return {"blocked": False, "model_sees": seen, "report": report.to_json()}
    request = {"model": body.model, "messages": [{"role": "user", "content": body.text}]}
    run = run_chat(app, c.session, c.tenant, request, source="playground")
    sees = run.protected[-1]["content"] if run.protected else None
    if run.status != 200:
        error = run.body.get("error", {}) if isinstance(run.body, dict) else {}
        return {
            "blocked": error.get("code", "").startswith("pii_blocked"),
            "model_sees": sees,
            "message": error.get("message", f"provider returned {run.status}"),
            "report": run.report.to_json(),
        }
    choices = run.body.get("choices") or [{}]
    reply = (choices[0].get("message") or {}).get("content")
    return {"blocked": False, "model_sees": sees, "reply": reply, "report": run.report.to_json()}
