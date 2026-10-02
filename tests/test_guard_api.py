"""MP Guard end to end over HTTP, against a fake AI provider that records what it receives.

All values are made up.
"""

from __future__ import annotations

import base64
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

    def __call__(self, request: httpx.Request) -> httpx.Response:
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
    g.call("GET", "/guard/settings", auditor)  # auditors can see the setup
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
