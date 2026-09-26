"""The platform end to end over HTTP: organizations, roles, the lifecycle, workers, and reports."""

from __future__ import annotations

import base64
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from model_passport.llm import synthetic
from model_passport.llm.entities import write_corpus
from model_passport.platform import services
from model_passport.platform.api.app import create_app
from model_passport.platform.db import ALL_TENANTS, make_engine, scoped_session, session_factory
from model_passport.platform.settings import Settings
from model_passport.platform.storage import open_store
from model_passport.platform.worker import WorkerContext, drain

pytest.importorskip("torch")
PASSWORD = "a long test password"


@dataclass
class Platform:
    client: TestClient
    worker: WorkerContext
    tmp: Path

    def login(self, email: str, tenant: str | None = None) -> dict[str, str]:
        response = self.client.post(
            "/api/v1/auth/login", json={"email": email, "password": PASSWORD}
        )
        assert response.status_code == 200, response.text
        headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
        if tenant:
            headers["X-MP-Tenant"] = tenant
        return headers

    def call(
        self, method: str, path: str, headers: dict[str, str], expect: int = 200, **kwargs: Any
    ) -> Any:
        response = self.client.request(method, f"/api/v1{path}", headers=headers, **kwargs)
        assert response.status_code == expect, f"{method} {path}: {response.text}"
        return (
            response.json()
            if response.content and "json" in response.headers.get("content-type", "")
            else response.text
        )


def _database(kind: str, tmp_path: Path) -> str:
    if kind == "sqlite":
        return f"sqlite:///{tmp_path}/mp.db"
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tempfile.mkdtemp())
    server.psql("CREATE ROLE mp_app LOGIN PASSWORD 'pw'; CREATE DATABASE mp OWNER mp_app;")
    host = server.get_uri().split("host=")[1]
    # A non-superuser role, as in production, so row-level security is enforced.
    return f"postgresql+psycopg://mp_app:pw@/mp?host={host}"


@pytest.fixture(params=["sqlite", "postgres"])
def platform(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Platform]:
    settings = Settings(
        database_url=_database(request.param, tmp_path),
        jwt_secret="s" * 48,
        master_key=base64.b64decode(base64.b64encode(os.urandom(32))),
        storage=f"file://{tmp_path}/objects",
        base_domain="mp.test",
    )
    app = create_app(settings)
    factory = session_factory(make_engine(settings.database_url))
    with scoped_session(factory, ALL_TENANTS) as session:
        services.create_user(session, "root@mp.test", PASSWORD, "Root", super_admin=True)
    worker = WorkerContext(
        factory=factory,
        store=open_store(settings.storage),
        master_key=settings.master_key,
        name="test-worker",
    )
    with TestClient(app) as client:
        yield Platform(client, worker, tmp_path)


def _setup_org(p: Platform) -> dict[str, dict[str, str]]:
    root = p.login("root@mp.test")
    for slug, name in (("usps", "USPS"), ("ups", "UPS")):
        p.call("POST", "/platform/tenants", root, 201, json={"slug": slug, "name": name})
        p.call(
            "POST",
            f"/platform/tenants/{slug}/admins",
            root,
            201,
            json={"email": f"admin@{slug}.test", "password": PASSWORD, "role": "org_admin"},
        )
    admin = p.login("admin@usps.test", "usps")
    people = {
        "eng": "ml_engineer",
        "auditor": "compliance_auditor",
        "reviewer": "external_reviewer",
        "consumer": "end_consumer",
    }
    for who, role in people.items():
        p.call(
            "POST",
            "/members",
            admin,
            201,
            json={"email": f"{who}@usps.test", "password": PASSWORD, "role": role},
        )
    logins = {who: p.login(f"{who}@usps.test", "usps") for who in people}
    return {"root": root, "admin": admin, "ups_admin": p.login("admin@ups.test", "ups"), **logins}


def _detect(p: Platform, who: dict[str, dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Phase 1: register data and model, train on the platform, audit."""
    eng, auditor = who["eng"], who["auditor"]
    corpus = p.tmp / "tickets.jsonl"
    write_corpus(synthetic.corpus(300, seed=3), corpus)
    provenance = {
        "name": "support-tickets",
        "source": "helpdesk export 2025",
        "license": "internal",
        "consent": "obtained",
    }
    dataset = p.call(
        "POST",
        "/datasets",
        eng,
        201,
        files={"file": ("tickets.jsonl", corpus.read_bytes())},
        data=provenance,
    )
    assert dataset["records"] == 300
    spec = {"name": "support-assistant", "access": "open-weight", "base": "tiny"}
    model = p.call("POST", "/models", eng, 201, json=spec)
    request = {
        "version": "1.0.0",
        "dataset_id": dataset["id"],
        "reference_dataset_id": dataset["id"],
        "train": True,
    }
    version = p.call("POST", f"/models/{model['id']}/versions", eng, 201, json=request)
    assert drain(p.worker) == 2  # train, then audit
    v1 = p.call("GET", f"/versions/{version['id']}", eng)
    assert v1["state"] == "findings"
    assert v1["critical"] > 0
    assert v1["verdict"] == "fail"
    findings = p.call("GET", f"/versions/{v1['id']}/findings", auditor)["audit"]["findings"]
    assert findings[0]["severity"] == "critical"
    return model, v1


def _access_rules(p: Platform, who: dict[str, dict[str, str]], v1: dict[str, Any]) -> None:
    reviewer_findings = f"/versions/{v1['id']}/findings"
    p.call("GET", reviewer_findings, who["reviewer"], 403)  # reviewers see reports, not findings
    p.call("GET", "/models", who["consumer"], 403)
    p.call("GET", "/models", who["ups_admin"] | {"X-MP-Tenant": "usps"}, 403)  # other tenant
    assert p.call("GET", "/models", who["ups_admin"]) == []  # UPS sees nothing of USPS
    p.call("POST", f"/versions/{v1['id']}/approvals", who["eng"], 403, json={"decision": "approve"})


def _remediate(
    p: Platform, eng: dict[str, str], model: dict[str, Any], v1: dict[str, Any]
) -> dict[str, Any]:
    """Phase 2: sanitize, retrain, and re-audit until the version is clean."""
    current = v1
    for _ in range(4):
        if current["state"] == "clean":
            break
        p.call("POST", f"/versions/{current['id']}/remediate", eng, 202)
        drain(p.worker)
        versions = p.call("GET", f"/models/{model['id']}", eng)["versions"]
        assert versions[-2]["state"] == "superseded"
        current = versions[-1]
    assert current["state"] == "clean", current
    assert current["critical"] == current["high"] == 0
    assert current["parent_id"] is not None
    return current


def _release_and_kill(p: Platform, who: dict[str, dict[str, str]], vid: str) -> None:
    """Phase 3: canary, verification, approval, release; then the kill switch."""
    eng, auditor, admin = who["eng"], who["auditor"], who["admin"]
    p.call("POST", f"/versions/{vid}/transition", eng, json={"to": "canary"})
    p.call("POST", f"/versions/{vid}/transition", eng, json={"to": "verifying"})
    p.call("POST", f"/versions/{vid}/transition", eng, 409, json={"to": "approved"})
    decision = {"decision": "approve", "comment": "re-audit clean"}
    p.call("POST", f"/versions/{vid}/approvals", auditor, json=decision)
    p.call("POST", f"/versions/{vid}/transition", eng, 409, json={"to": "released"})
    released = p.call("POST", f"/versions/{vid}/transition", admin, json={"to": "released"})
    assert released["state"] == "released"
    killed = p.call("POST", f"/versions/{vid}/kill", auditor, json={"reason": "leak reported"})
    assert killed["state"] == "killed"


def _evidence(p: Platform, who: dict[str, dict[str, str]], model: dict[str, Any], vid: str) -> None:
    """What investors, acquirers, and auditors receive."""
    reviewer, auditor = who["reviewer"], who["auditor"]
    report = p.call("GET", f"/models/{model['id']}/diligence", reviewer)
    assert report["summary"]["all_attestations_verified"] is True
    assert report["audit_log"]["intact"] is True
    assert report["datasets"][0]["consent"] == "obtained"
    assert len(report["versions"]) >= 2
    html = p.call("GET", f"/models/{model['id']}/diligence?format=html", reviewer)
    assert "Diligence report: support-assistant" in html
    attestation = p.call("GET", f"/versions/{vid}/attestation", auditor)
    assert attestation["attestation"]["parent"]["attestation_sha256"]
    assert p.call("GET", "/audit-log/verify", auditor)["intact"] is True
    actions = {e["action"] for e in p.call("GET", "/audit-log?limit=1000", auditor)}
    steps = {
        "dataset.added",
        "version.auditing",
        "version.findings",
        "version.superseded",
        "version.clean",
        "version.approved",
        "version.released",
        "version.killed",
    }
    assert steps <= actions
    assert p.call("GET", "/dashboard", who["admin"])["versions_by_state"]["killed"] == 1
    assert p.call("GET", "/analytics", auditor)["remediation_rounds"] >= 1
    p.call("POST", f"/models/{model['id']}/reports", reviewer, 202)
    assert drain(p.worker) == 1


def test_the_full_product_flow(platform: Platform) -> None:
    who = _setup_org(platform)
    model, v1 = _detect(platform, who)
    _access_rules(platform, who, v1)
    clean = _remediate(platform, who["eng"], model, v1)
    _release_and_kill(platform, who, clean["id"])
    _evidence(platform, who, model, clean["id"])


def test_sign_in_and_tenant_resolution(platform: Platform) -> None:
    p = platform
    p.call("POST", "/auth/login", {}, 401, json={"email": "root@mp.test", "password": "nope"})
    p.call("GET", "/me", {}, 401)
    who = _setup_org(p)
    me = p.call("GET", "/me", who["eng"])
    assert me["memberships"] == [{"tenant": "usps", "role": "ml_engineer"}]
    no_tenant = {"Authorization": who["eng"]["Authorization"]}
    p.call("GET", "/models", no_tenant, 400)
    by_host = no_tenant | {"Host": "usps.mp.test"}
    assert p.call("GET", "/models", by_host) == []
    p.call("GET", "/platform/tenants", who["admin"], 403)  # org admins are not super admins
    root = who["root"]
    assert {t["slug"] for t in p.call("GET", "/platform/tenants", root)} == {"usps", "ups"}
    p.call("PATCH", "/platform/tenants/ups", root, json={"status": "suspended"})
    p.call("GET", "/models", who["ups_admin"], 403)  # suspended organizations are locked
    assert p.call("GET", "/platform/audit-log/verify", root)["intact"] is True
    assert p.client.get("/health").json()["status"] == "ok"


def test_canary_testers_can_block_approval(platform: Platform) -> None:
    from model_passport.platform.db import ALL_TENANTS, scoped_session
    from model_passport.platform.models import ModelVersion, Role, State

    p = platform
    who = _setup_org(p)
    p.call(
        "POST",
        "/members",
        who["admin"],
        201,
        json={"email": "tester@usps.test", "password": PASSWORD, "role": "canary_tester"},
    )
    tester = p.login("tester@usps.test", "usps")
    corpus = p.tmp / "c.jsonl"
    write_corpus(synthetic.corpus(20, seed=1), corpus)
    dataset = p.call(
        "POST",
        "/datasets",
        who["eng"],
        201,
        files={"file": ("c.jsonl", corpus.read_bytes())},
        data={"name": "d"},
    )
    model = p.call(
        "POST",
        "/models",
        who["eng"],
        201,
        json={"name": "m", "access": "api", "base": "anthropic:claude-sonnet-5"},
    )
    version = p.call(
        "POST",
        f"/models/{model['id']}/versions",
        who["eng"],
        201,
        json={"reference_dataset_id": dataset["id"]},
    )
    with scoped_session(p.worker.factory, ALL_TENANTS) as s:
        row = s.get(ModelVersion, version["id"])
        assert row is not None
        row.state = State.CLEAN  # as if a clean audit had just finished
    vid = version["id"]
    p.call(
        "POST",
        f"/versions/{vid}/test-reports",
        tester,
        409,
        json={"prompt_count": 5, "leaks_found": 0},
    )
    p.call("POST", f"/versions/{vid}/transition", who["eng"], json={"to": "canary"})
    p.call(
        "POST",
        f"/versions/{vid}/test-reports",
        who["eng"],
        403,
        json={"prompt_count": 1, "leaks_found": 0},
    )
    p.call(
        "POST",
        f"/versions/{vid}/test-reports",
        tester,
        201,
        json={"prompt_count": 40, "leaks_found": 1, "summary": "prefix attack surfaced an email"},
    )
    p.call("POST", f"/versions/{vid}/transition", who["eng"], json={"to": "verifying"})
    blocked = p.client.post(
        f"/api/v1/versions/{vid}/approvals", headers=who["auditor"], json={"decision": "approve"}
    )
    assert blocked.status_code == 409
    assert "canary testers reported leaks" in blocked.text
    reports = p.call("GET", f"/versions/{vid}/test-reports", who["auditor"])
    assert reports[0]["leaks_found"] == 1
    assert Role.CANARY_TESTER.value == "canary_tester"
