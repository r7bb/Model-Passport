"""MP Guard end to end over HTTP, against a fake AI provider that records what it receives.

All values are made up.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from model_passport.core.schema import EntityAuditResult, EntityFinding, Severity
from model_passport.guard.engine import fingerprint
from model_passport.platform import guard as service
from model_passport.platform import services
from model_passport.platform.api.app import create_app
from model_passport.platform.auditlog import Actor
from model_passport.platform.db import ALL_TENANTS, make_engine, scoped_session, session_factory
from model_passport.platform.guard import GuardError, MemorizedCache, check_upstream
from model_passport.platform.models import Access, Dataset, GuardSettings, Tenant
from model_passport.platform.settings import Settings

PASSWORD = "a long test password"
EMAIL, SSN, CARD = "jane.doe@example.com", "123-45-6789", "4111 1111 1111 1111"
MEMORIZED = "leaked.patient@example.com"
PROVIDER_KEY = "sk-provider-test-0001"


@dataclass
class FakeProvider:
    """Answers chat completions; ``reply`` may use the placeholders it was sent."""

    reply: str = "Done."
    received: list[dict[str, Any]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.received.append(json.loads(request.content))
        self.headers.append(dict(request.headers))
        message = {"role": "assistant", "content": self.reply}
        return httpx.Response(
            200, json={"id": "cmpl-1", "choices": [{"index": 0, "message": message}]}
        )


@dataclass
class Guarded:
    client: TestClient
    provider: FakeProvider
    factory: Any
    master: bytes

    def login(self, email: str, org: str | None = "acme") -> dict[str, str]:
        response = self.client.post(
            "/api/v1/auth/login", json={"email": email, "password": PASSWORD}
        )
        assert response.status_code == 200, response.text
        headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
        return headers | ({"X-MP-Tenant": org} if org else {})

    def call(
        self, method: str, path: str, headers: dict[str, str], expect: int = 200, **kw: Any
    ) -> Any:
        response = self.client.request(method, f"/api/v1{path}", headers=headers, **kw)
        assert response.status_code == expect, f"{method} {path}: {response.text}"
        return response.json() if response.content else None

    def chat(self, key: str, content: str, expect: int = 200, **extra: Any) -> httpx.Response:
        body = {"messages": [{"role": "user", "content": content}], **extra}
        response = self.client.post(
            "/guard/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
        )
        assert response.status_code == expect, response.text
        return response


def _database(kind: str, tmp_path: Path) -> str:
    if kind == "sqlite":
        return f"sqlite:///{tmp_path}/mp.db"
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tempfile.mkdtemp())
    server.psql("CREATE ROLE mp_app LOGIN PASSWORD 'pw'; CREATE DATABASE mp OWNER mp_app;")
    host = server.get_uri().split("host=")[1]
    return f"postgresql+psycopg://mp_app:pw@/mp?host={host}"


@pytest.fixture(params=["sqlite", "postgres"])
def guarded(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Guarded]:
    master = base64.b64decode(base64.b64encode(os.urandom(32)))
    settings = Settings(
        database_url=_database(request.param, tmp_path),
        jwt_secret="g" * 48,
        master_key=master,
        storage=f"file://{tmp_path}/objects",
        guard_private_upstreams=True,  # the fake provider has no public address
    )
    app = create_app(settings)
    provider = FakeProvider()
    app.state.mp.guard_http = httpx.Client(transport=httpx.MockTransport(provider))
    app.state.mp.memorized = MemorizedCache(ttl=0)
    factory = session_factory(make_engine(settings.database_url))
    with scoped_session(factory, ALL_TENANTS) as session:
        services.create_user(session, "root@mp.test", PASSWORD, "Root", super_admin=True)
    with TestClient(app) as client:
        g = Guarded(client, provider, factory, master)
        root = g.login("root@mp.test", None)
        for slug in ("acme", "globex"):
            g.call(
                "POST", "/platform/tenants", root, 201, json={"slug": slug, "name": slug.title()}
            )
            admin = {"email": f"admin@{slug}.test", "password": PASSWORD, "role": "org_admin"}
            g.call("POST", f"/platform/tenants/{slug}/admins", root, 201, json=admin)
        admin = g.login("admin@acme.test")
        for who, role in (
            ("eng", "ml_engineer"),
            ("auditor", "compliance_auditor"),
            ("user", "end_consumer"),
        ):
            body = {"email": f"{who}@acme.test", "password": PASSWORD, "role": role}
            g.call("POST", "/members", admin, 201, json=body)
        yield g


def _configure(
    g: Guarded, who: dict[str, str], policy: dict[str, Any] | None = None
) -> dict[str, Any]:
    body = {
        "upstream_url": "https://provider.test/v1",
        "upstream_key": PROVIDER_KEY,
        "default_model": "gpt-test",
        "policy": policy or {},
    }
    return g.call("PUT", "/guard/settings", who, json=body)  # type: ignore[no-any-return]


def test_requests_are_protected_and_replies_restored(guarded: Guarded) -> None:
    g, eng = guarded, guarded.login("eng@acme.test")
    out = _configure(g, eng)
    assert out["has_upstream_key"] is True
    assert PROVIDER_KEY not in json.dumps(out)
    key = g.call("POST", "/guard/keys", eng, 201, json={"name": "support-bot"})["key"]
    assert key.startswith("mpk_")

    g.provider.reply = f"I will email [EMAIL_1]. The card on file is {CARD}."
    response = g.chat(key, f"My email is {EMAIL} and my SSN is {SSN}. Update my account.")
    sent = json.dumps(g.provider.received[-1])
    assert EMAIL not in sent  # the model never saw the values
    assert SSN not in sent
    assert "[EMAIL_1]" in sent
    assert "[SSN_1]" in sent
    assert g.provider.received[-1]["model"] == "gpt-test"
    assert g.provider.headers[-1]["authorization"] == f"Bearer {PROVIDER_KEY}"
    reply = response.json()["choices"][0]["message"]["content"]
    assert reply.startswith(f"I will email {EMAIL}.")  # restored for the user
    assert CARD not in reply  # a card the model produced is redacted
    assert "masked=2" in response.headers["x-mp-guard"]

    listed = g.call("GET", "/guard/keys", eng)
    assert "key" not in listed[0]
    assert listed[0]["last_used_at"] is not None


def test_keys_policies_and_errors(guarded: Guarded) -> None:
    g, eng = guarded, guarded.login("eng@acme.test")
    _configure(g, eng, {"inbound": {"SSN": "block"}})
    key = g.call("POST", "/guard/keys", eng, 201, json={"name": "intake"})
    calls = len(g.provider.received)
    blocked = g.chat(key["key"], f"SSN {SSN}", expect=400)
    assert blocked.json()["error"]["code"] == "pii_blocked"
    assert len(g.provider.received) == calls  # never forwarded
    assert (
        g.chat(key["key"], "hi", expect=400, stream=True).json()["error"]["code"]
        == "stream_unsupported"
    )
    bad = g.chat("mpk_nothing", "hi", expect=401).json()
    assert bad["error"]["code"] == "invalid_api_key"
    g.call("DELETE", f"/guard/keys/{key['id']}", eng, 204)
    assert "revoked" in g.chat(key["key"], "hi", expect=401).json()["error"]["message"]


def test_roles_and_isolation(guarded: Guarded) -> None:
    g = guarded
    eng, auditor, user = (
        g.login("eng@acme.test"),
        g.login("auditor@acme.test"),
        g.login("user@acme.test"),
    )
    _configure(g, eng)
    g.call("POST", "/guard/keys", eng, 201, json={"name": "app"})
    seen = g.call("GET", "/guard/settings", auditor)  # auditors can see the setup
    assert seen["effective"]["SSN"] == {"request": "mask", "reply": "redact"}
    assert seen["effective"]["DATE"] == {"request": "allow", "reply": "allow"}
    g.call("PUT", "/guard/settings", auditor, 403, json={"upstream_url": "https://x.test/v1"})
    g.call("GET", "/guard/keys", user, 403)
    other = g.login("admin@globex.test", "globex")
    assert g.call("GET", "/guard/keys", other) == []
    assert g.call("GET", "/guard/settings", other)["has_upstream_key"] is False


def test_the_provider_key_is_encrypted_at_rest(guarded: Guarded) -> None:
    g = guarded
    _configure(g, g.login("eng@acme.test"))
    with scoped_session(g.factory, ALL_TENANTS) as session:
        stored = session.scalars(select(GuardSettings)).one()
        assert stored.upstream_key is not None
        assert PROVIDER_KEY.encode() not in stored.upstream_key
        tenant = session.get(Tenant, stored.tenant_id)
        assert tenant is not None
        assert service.provider_key(stored, service.data_key(g.master, tenant)) == PROVIDER_KEY


def test_a_new_provider_host_needs_the_key_again(guarded: Guarded) -> None:
    """Otherwise anyone who manages the guard could send the saved key to a host they own."""
    g, eng = guarded, guarded.login("eng@acme.test")
    _configure(g, eng)
    keep = {"upstream_key": None, "default_model": "gpt-test", "policy": {}}
    moved = {**keep, "upstream_url": "https://attacker.test/v1"}
    error = g.call("PUT", "/guard/settings", eng, 400, json=moved)
    assert "API key" in error["detail"]
    same_host = {**keep, "upstream_url": "https://provider.test/v2"}
    assert g.call("PUT", "/guard/settings", eng, json=same_host)["has_upstream_key"] is True
    rekeyed = {**moved, "upstream_key": "sk-new"}
    assert g.call("PUT", "/guard/settings", eng, json=rekeyed)["upstream_url"].startswith(
        "https://attacker.test"
    )


def test_values_the_audit_found_memorized_are_redacted(guarded: Guarded) -> None:
    g, eng = guarded, guarded.login("eng@acme.test")
    _configure(g, eng)
    with scoped_session(g.factory, ALL_TENANTS) as session:
        tenant = session.scalars(select(Tenant).where(Tenant.slug == "acme")).one()
        key = service.fingerprint_key(service.data_key(g.master, tenant))
        finding = EntityFinding(
            record="r1",
            span=(0, len(MEMORIZED)),
            entity_type="EMAIL",
            masked_value="l***@e***.com",
            fingerprint=fingerprint(key, "EMAIL", MEMORIZED),
            score=5.0,
            p_value=1e-6,
            q_value=1e-5,
            likelihood=1.0,
            impact=0.8,
            risk=8.0,
            severity=Severity.HIGH,
            confirmed=True,
        )
        audit = EntityAuditResult(
            model="m", access="logprobs", primary_method="loss", methods=["loss"],
            references=5, entities_audited=1, controls=1, findings=[finding],
        )  # fmt: skip
        actor = Actor(None, "test")
        dataset = Dataset(tenant_id=tenant.id, name="tickets", sha256="0" * 64, object_key="k")
        session.add(dataset)
        session.flush()
        model = services.register_model(session, tenant, "bot", Access.API, "openai:x", "", actor)
        version = services.register_version(session, model, "1.0.0", dataset, dataset, actor)
        version.audit = audit.model_dump(mode="json")
    api_key = g.call("POST", "/guard/keys", eng, 201, json={"name": "bot"})["key"]
    g.provider.reply = f"Try {MEMORIZED} or help@example.com."
    reply = g.chat(api_key, "Who should I contact?").json()["choices"][0]["message"]["content"]
    assert MEMORIZED not in reply
    assert "help@example.com" in reply  # an ordinary value passes
    event = g.call("GET", "/guard/events", eng)[0]
    assert event["report"]["memorized"] == {"EMAIL": 1}


def test_activity_playground_and_scan_hold_no_values(guarded: Guarded) -> None:
    g, eng = guarded, guarded.login("eng@acme.test")
    _configure(g, eng)
    key = g.call("POST", "/guard/keys", eng, 201, json={"name": "agent"})["key"]
    g.chat(key, f"Contact {EMAIL}")
    preview = g.call("POST", "/guard/playground", eng, json={"text": f"Call me at {EMAIL}"})
    assert preview["model_sees"] == "Call me at [EMAIL_1]"
    g.provider.reply = "Sure, [EMAIL_1]."
    sent = g.call("POST", "/guard/playground", eng, json={"text": f"Hi, I'm {EMAIL}", "send": True})
    assert sent["reply"] == f"Sure, {EMAIL}."
    scanned = g.client.post(
        "/guard/v1/scan",
        json={"text": f"Tool result: SSN {SSN}", "direction": "reply"},
        headers={"Authorization": f"Bearer {key}"},
    ).json()
    assert SSN not in scanned["text"]
    stats = g.call("GET", "/guard/stats", eng)
    assert stats["requests"] == 3  # the API call, the sent playground run, and the scan
    assert stats["values"]["masked"] >= 2
    events = g.call("GET", "/guard/events", eng)
    log = g.call("GET", "/audit-log?limit=1000", g.login("auditor@acme.test"))
    recorded = json.dumps([events, stats, log])
    for value in (EMAIL, SSN, PROVIDER_KEY, key):
        assert value not in recorded
    actions = {e["action"] for e in log}
    assert {"guard.key.created", "guard.settings.updated"} <= actions
    assert g.call("GET", "/audit-log/verify", g.login("auditor@acme.test"))["intact"] is True


def test_provider_addresses_on_private_networks_are_refused() -> None:
    for url in ("https://127.0.0.1/v1", "https://169.254.169.254/latest", "https://10.0.0.5/v1"):
        with pytest.raises(GuardError, match="private or local"):
            check_upstream(url)
    with pytest.raises(GuardError, match="https"):
        check_upstream("http://8.8.8.8/v1")
    with pytest.raises(GuardError, match="full URL"):
        check_upstream("api.openai.com")
    assert (
        check_upstream("http://localhost:11434/v1/", allow_private=True)
        == "http://localhost:11434/v1"
    )


# --- The provider address is checked again on every request, and the connection is pinned ----

PUBLIC_IP = "93.184.216.34"


class FakeDns:
    """Stands in for name resolution; ``answers`` can change between calls."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)

    def __call__(self, host: str, port: int) -> list[str]:
        return list(self.answers)


def _strict(g: Guarded, monkeypatch: pytest.MonkeyPatch, dns: FakeDns) -> dict[str, str]:
    """Turn the private-network allowance off, as in production, and save the settings."""
    state = g.client.app.state.mp  # type: ignore[attr-defined]
    state.settings = dataclasses.replace(state.settings, guard_private_upstreams=False)
    monkeypatch.setattr(service, "resolve_host", dns)
    admin = g.login("admin@acme.test")
    _configure(g, admin)
    return g.call("POST", "/guard/keys", admin, 201, json={"name": "k"})  # type: ignore[no-any-return]


@pytest.mark.parametrize("rebound", ["127.0.0.1", "169.254.169.254", "::1", "10.0.0.5"])
def test_a_host_that_later_resolves_privately_is_refused_and_never_called(
    guarded: Guarded, monkeypatch: pytest.MonkeyPatch, rebound: str
) -> None:
    dns = FakeDns(PUBLIC_IP)
    key = _strict(guarded, monkeypatch, dns)["key"]
    dns.answers = [rebound]
    reply = guarded.chat(key, "hello", expect=502)
    assert reply.json()["error"]["code"] == "provider_blocked"
    assert rebound not in reply.text
    assert guarded.provider.requests == []


def test_a_mixed_public_and_private_answer_is_refused(
    guarded: Guarded, monkeypatch: pytest.MonkeyPatch
) -> None:
    dns = FakeDns(PUBLIC_IP)
    key = _strict(guarded, monkeypatch, dns)["key"]
    dns.answers = [PUBLIC_IP, "127.0.0.1"]
    guarded.chat(key, "hello", expect=502)
    assert guarded.provider.requests == []


def test_a_host_that_stops_resolving_is_refused(
    guarded: Guarded, monkeypatch: pytest.MonkeyPatch
) -> None:
    dns = FakeDns(PUBLIC_IP)
    key = _strict(guarded, monkeypatch, dns)["key"]
    dns.answers = []
    guarded.chat(key, "hello", expect=502)
    assert guarded.provider.requests == []


@pytest.mark.parametrize(
    ("address", "netloc"), [(PUBLIC_IP, PUBLIC_IP), ("2606:4700::1111", "[2606:4700::1111]")]
)
def test_the_request_goes_to_the_checked_address_with_the_real_name(
    guarded: Guarded, monkeypatch: pytest.MonkeyPatch, address: str, netloc: str
) -> None:
    key = _strict(guarded, monkeypatch, FakeDns(address))["key"]
    guarded.chat(key, "hello")
    (request,) = guarded.provider.requests
    assert request.url.scheme == "https"
    assert request.url.netloc.decode() == netloc
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["host"] == "provider.test"
    assert request.extensions["sni_hostname"] == "provider.test"
    assert request.headers["authorization"] == f"Bearer {PROVIDER_KEY}"


def test_a_private_upstream_is_still_allowed_when_the_operator_permits_it(
    guarded: Guarded, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "resolve_host", FakeDns("127.0.0.1"))
    key = _configure_key(guarded)
    guarded.chat(key, "hello")
    assert len(guarded.provider.requests) == 1


def _configure_key(g: Guarded) -> str:
    admin = g.login("admin@acme.test")
    _configure(g, admin)
    return g.call("POST", "/guard/keys", admin, 201, json={"name": "k"})["key"]  # type: ignore[no-any-return]


# --- Limits on the proxy: body size, max_tokens, n, and request rates -------------------------


def _limited(g: Guarded, **limits: int) -> str:
    """Configure the org, apply small limits to the running app, and return a client key."""
    eng = g.login("eng@acme.test")
    _configure(g, eng)
    state = g.client.app.state.mp  # type: ignore[attr-defined]
    state.settings = dataclasses.replace(state.settings, **limits)
    return str(g.call("POST", "/guard/keys", eng, 201, json={"name": "limited"})["key"])


def _new_key(g: Guarded, name: str) -> str:
    return str(
        g.call("POST", "/guard/keys", g.login("eng@acme.test"), 201, json={"name": name})["key"]
    )


def _post(g: Guarded, path: str, key: str, **kw: Any) -> httpx.Response:
    return g.client.post(f"/guard/v1{path}", headers={"Authorization": f"Bearer {key}"}, **kw)


def test_body_over_the_cap_is_refused_before_forwarding(guarded: Guarded) -> None:
    g = guarded
    key = _limited(g, guard_max_body_bytes=2000)
    big = {"messages": [{"role": "user", "content": "x" * 5000}]}
    sent = len(g.provider.received)

    declared = _post(g, "/chat/completions", key, json=big)
    assert declared.status_code == 413
    assert declared.json()["error"]["code"] == "request_too_large"

    def chunks() -> Iterator[bytes]:  # no Content-Length: the limit applies to the bytes read
        yield b'{"messages": [{"role": "user", "content": "'
        yield b"x" * 5000
        yield b'"}]}'

    streamed = _post(g, "/chat/completions", key, content=chunks())
    assert streamed.status_code == 413
    assert _post(g, "/scan", key, json={"text": "x" * 5000}).status_code == 413
    assert len(g.provider.received) == sent  # never forwarded
    assert g.chat(key, "hi").status_code == 200  # a small body still works


def test_malformed_body_gets_an_openai_style_error(guarded: Guarded) -> None:
    g = guarded
    key = _limited(g)
    for content in (b"not json", b"[1, 2]"):
        response = _post(g, "/chat/completions", key, content=content)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"


def test_max_tokens_over_the_cap_is_rejected_and_absent_is_capped(guarded: Guarded) -> None:
    g = guarded
    key = _limited(g, guard_max_tokens=100)
    sent = len(g.provider.received)
    for field_name in ("max_tokens", "max_completion_tokens"):
        refused = g.chat(key, "hi", expect=422, **{field_name: 101})
        assert refused.json()["error"]["code"] == "invalid_request"
    for bad in (0, -5, True, "50"):
        g.chat(key, "hi", expect=422, max_tokens=bad)
    assert len(g.provider.received) == sent

    g.chat(key, "hi", temperature=0.2)
    assert g.provider.received[-1]["max_tokens"] == 100  # no provider default left unbounded
    assert g.provider.received[-1]["temperature"] == 0.2  # the rest is untouched
    g.chat(key, "hi", max_tokens=100)
    assert g.provider.received[-1]["max_tokens"] == 100
    g.chat(key, "hi", max_completion_tokens=40)
    assert g.provider.received[-1]["max_completion_tokens"] == 40
    assert "max_tokens" not in g.provider.received[-1]


def test_n_above_the_cap_is_rejected(guarded: Guarded) -> None:
    g = guarded
    key = _limited(g, guard_max_n=2)
    sent = len(g.provider.received)
    assert g.chat(key, "hi", expect=422, n=3).json()["error"]["code"] == "invalid_request"
    g.chat(key, "hi", expect=422, n="2")
    assert len(g.provider.received) == sent
    g.chat(key, "hi", n=2)
    assert g.provider.received[-1]["n"] == 2


def test_per_key_rate_limit_returns_429_with_retry_after(guarded: Guarded) -> None:
    g = guarded
    first = _limited(g, guard_rate_per_key=2, guard_rate_per_ip=100)
    second = _new_key(g, "other")
    g.chat(first, "hi")
    g.chat(first, "hi")
    limited = g.chat(first, "hi", expect=429)
    assert int(limited.headers["retry-after"]) >= 1
    assert limited.json()["error"]["code"] == "rate_limited"
    assert _post(g, "/scan", first, json={"text": "hi"}).status_code == 429  # shared by routes
    g.chat(second, "hi")  # another key is unaffected


def test_per_ip_rate_limit_applies_across_keys(guarded: Guarded) -> None:
    g = guarded
    first = _limited(g, guard_rate_per_key=100, guard_rate_per_ip=2)
    second = _new_key(g, "other")
    g.chat(first, "hi")
    g.chat(second, "hi")
    limited = g.chat(second, "hi", expect=429)
    assert int(limited.headers["retry-after"]) >= 1
    spoofed = {"Authorization": f"Bearer {first}", "X-Forwarded-For": "203.0.113.9"}
    response = g.client.post("/guard/v1/scan", headers=spoofed, json={"text": "hi"})
    assert response.status_code == 429  # X-Forwarded-For is not trusted


def test_rate_limiter_windows_and_bounded_memory() -> None:
    from model_passport.platform.security import RateLimiter

    limiter = RateLimiter(limit=2, window=60, max_keys=3)
    assert [limiter.check("a", now=0.0) for _ in range(2)] == [0, 0]
    assert limiter.check("a", now=10.0) == 51
    assert limiter.check("a", now=60.0) == 0  # a new window
    for index in range(50):
        limiter.check(f"key-{index}", now=61.0)
    assert len(limiter) <= 3
    assert RateLimiter(limit=0).check("a") == 0  # a limit of 0 turns it off


def test_deeply_nested_body_inside_the_cap_is_a_400(guarded: Guarded) -> None:
    g = guarded
    key = _limited(g)
    response = _post(g, "/chat/completions", key, content=b"[" * 100_000)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MP_GUARD_RATE_PER_KEY", "-1"),
        ("MP_GUARD_RATE_PER_IP", "-1"),
        ("MP_GUARD_MAX_BODY_BYTES", "0"),
        ("MP_GUARD_MAX_TOKENS", "0"),
        ("MP_GUARD_MAX_N", "0"),
        ("MP_GUARD_MAX_N", "-3"),
    ],
)
def test_unsafe_guard_limits_are_refused(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    from model_passport.platform.settings import SettingsError

    monkeypatch.setenv("MP_JWT_SECRET", "j" * 40)
    monkeypatch.setenv("MP_MASTER_KEY", base64.b64encode(os.urandom(32)).decode())
    monkeypatch.setenv(name, value)
    with pytest.raises(SettingsError, match=name):
        Settings.from_env()


def test_zero_rate_limits_stay_valid_as_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MP_JWT_SECRET", "j" * 40)
    monkeypatch.setenv("MP_MASTER_KEY", base64.b64encode(os.urandom(32)).decode())
    monkeypatch.setenv("MP_GUARD_RATE_PER_KEY", "0")
    monkeypatch.setenv("MP_GUARD_RATE_PER_IP", "0")
    settings = Settings.from_env()
    assert (settings.guard_rate_per_key, settings.guard_rate_per_ip) == (0, 0)
