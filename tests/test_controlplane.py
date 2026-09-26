"""The Go control plane against a real PostgreSQL, driven from Python over gRPC.

Checks roles and lifecycle rules, deploy/rollback/kill, and that the Go service appends to the
same hash-chained audit log that Python verifies.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from model_passport.platform import auditlog, services
from model_passport.platform.db import (
    ALL_TENANTS,
    make_engine,
    migrate,
    scoped_session,
    session_factory,
)
from model_passport.platform.models import Access, Dataset, ModelVersion, Role, State
from model_passport.platform.security import issue_token

pgserver = pytest.importorskip("pgserver")
pytest.importorskip("grpc")
from model_passport.platform.controlplane import ControlPlane, ControlPlaneError  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SECRET = "c" * 48
MASTER = os.urandom(32)
ADMIN = auditlog.Actor("seed", "seed@mp.test")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class Cluster:
    address: str
    factory: object
    tenant_id: str
    tokens: dict[str, str]
    ids: dict[str, str]
    database_url: str

    def as_(self, who: str) -> ControlPlane:
        return ControlPlane(self.address, self.tokens[who], "usps")


def _seed(factory: object) -> tuple[str, dict[str, str], dict[str, str]]:
    with scoped_session(factory, ALL_TENANTS) as s:  # type: ignore[arg-type]
        tenant = services.create_tenant(s, MASTER, "usps", "USPS", ADMIN)
        tokens = {}
        for who, role in [
            ("admin", Role.ORG_ADMIN),
            ("eng", Role.ML_ENGINEER),
            ("auditor", Role.COMPLIANCE_AUDITOR),
            ("tester", Role.CANARY_TESTER),
            ("consumer", Role.END_CONSUMER),
        ]:
            user = services.create_user(s, f"{who}@usps.test", "a long enough password")
            services.add_member(s, tenant, user, role, ADMIN)
            tokens[who] = issue_token(user.id, False, SECRET, 30)
        outsider = services.create_user(s, "x@ups.test", "a long enough password")
        tokens["outsider"] = issue_token(outsider.id, False, SECRET, 30)
        dataset = Dataset(tenant_id=tenant.id, name="d", sha256="0" * 64, object_key="k")
        s.add(dataset)
        s.flush()
        model = services.register_model(
            s, tenant, "assistant", Access.OPEN_WEIGHT, "tiny", "", ADMIN
        )
        ids = {"model": model.id}
        for version, state in [
            ("1.0.0", State.APPROVED),
            ("1.1.0", State.APPROVED),
            ("1.2.0", State.CLEAN),
            ("1.3.0", State.FINDINGS),
            ("2.0.0", State.CLEAN),
            ("2.1.0", State.CLEAN),
        ]:
            row = services.register_version(s, model, version, dataset, dataset, ADMIN)
            row.state = state
            ids[version] = row.id
        return tenant.id, tokens, ids


@pytest.fixture(scope="module")
def cluster(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Cluster]:
    go = shutil.which("go") or str(Path.home() / ".local/bin/go")
    if not Path(go).exists():
        pytest.skip("Go is not installed")
    build = tmp_path_factory.mktemp("bin")
    binary = build / "mp-controlplane"
    subprocess.run(
        [go, "build", "-o", str(binary), "./cmd/mp-controlplane"],
        cwd=REPO / "controlplane",
        check=True,
    )
    server = pgserver.get_server(tempfile.mkdtemp())
    server.psql("CREATE ROLE mp_app LOGIN PASSWORD 'pw'; CREATE DATABASE mp OWNER mp_app;")
    host = server.get_uri().split("host=")[1]
    url = f"postgresql+psycopg://mp_app:pw@/mp?host={host}"
    engine = make_engine(url)
    migrate(engine)
    factory = session_factory(engine)
    tenant_id, tokens, ids = _seed(factory)
    port = _free_port()
    env = os.environ | {
        "MP_DATABASE_URL": f"postgresql://mp_app:pw@/mp?host={host}",
        "MP_JWT_SECRET": SECRET,
        "MP_CONTROLPLANE_ADDR": f"127.0.0.1:{port}",
    }
    process = subprocess.Popen([str(binary)], env=env, stderr=subprocess.PIPE, text=True)
    for _ in range(100):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.1)
    else:
        process.kill()
        pytest.fail(
            f"control plane did not start: {process.stderr.read() if process.stderr else ''}"
        )
    yield Cluster(f"127.0.0.1:{port}", factory, tenant_id, tokens, ids, url)
    process.terminate()
    process.wait(timeout=10)


def _state(c: Cluster, version: str) -> State:
    with scoped_session(c.factory, c.tenant_id) as s:  # type: ignore[arg-type]
        row = s.get(ModelVersion, c.ids[version])
        assert row is not None
        return row.state


def test_roles_and_lifecycle_rules(cluster: Cluster) -> None:
    c = cluster
    with pytest.raises(ControlPlaneError) as denied:
        c.as_("eng").deploy(c.ids["1.0.0"], "consumer")  # releases need an org admin
    assert denied.value.code == "PERMISSION_DENIED"
    with pytest.raises(ControlPlaneError) as outsider:
        c.as_("outsider").deployments()
    assert outsider.value.code == "PERMISSION_DENIED"
    with pytest.raises(ControlPlaneError) as early:
        c.as_("eng").deploy(c.ids["1.3.0"], "dev")  # findings must be remediated first
    assert early.value.code == "FAILED_PRECONDITION"
    with pytest.raises(ControlPlaneError) as unapproved:
        c.as_("admin").deploy(c.ids["1.2.0"], "consumer")  # not approved yet
    assert unapproved.value.code == "FAILED_PRECONDITION"
    with pytest.raises(ControlPlaneError) as forged:
        ControlPlane(c.address, "forged", "usps").deployments()
    assert forged.value.code == "UNAUTHENTICATED"


def test_deploy_rollback_and_kill(cluster: Cluster) -> None:
    c = cluster
    dev = c.as_("tester").deploy(c.ids["1.2.0"], "dev")
    assert dev["endpoint"] == "/v1/dev/assistant"
    first = c.as_("admin").deploy(c.ids["1.0.0"], "consumer")
    second = c.as_("admin").deploy(c.ids["1.1.0"], "consumer")
    assert c.as_("consumer").resolve("assistant", "consumer")["version"] == "1.1.0"
    with pytest.raises(ControlPlaneError):
        c.as_("consumer").resolve("assistant", "dev")  # consumers cannot reach dev endpoints
    statuses = {d["id"]: d["status"] for d in c.as_("eng").deployments(c.ids["model"])}
    assert statuses[first["id"]] == "superseded"
    assert statuses[second["id"]] == "active"

    restored = c.as_("admin").rollback(c.ids["model"])
    assert restored["version"] == "1.0.0"
    assert c.as_("consumer").resolve("assistant", "consumer")["version"] == "1.0.0"

    with pytest.raises(ControlPlaneError) as no_reason:
        c.as_("auditor").kill(c.ids["1.0.0"], "")
    assert no_reason.value.code == "INVALID_ARGUMENT"
    killed = c.as_("auditor").kill(c.ids["1.0.0"], "customer data leaked in a reply")
    assert killed == {"deployments_stopped": 1, "version_state": "killed"}
    assert _state(c, "1.0.0") is State.KILLED
    assert c.as_("consumer").resolve("assistant", "consumer")["found"] is False
    with pytest.raises(ControlPlaneError) as blocked:
        c.as_("admin").deploy(c.ids["1.0.0"], "consumer")
    assert "kill switch" in str(blocked.value)
    assert c.as_("auditor").kill(c.ids["1.0.0"], "again")["deployments_stopped"] == 0  # idempotent


def test_go_and_python_share_one_verifiable_audit_log(cluster: Cluster) -> None:
    with scoped_session(cluster.factory, cluster.tenant_id) as s:  # type: ignore[arg-type]
        events = auditlog.chain(s, cluster.tenant_id)
        actions = [e.action for e in events]
        assert "deployment.created" in actions  # written by Go
        assert "version.registered" in actions  # written by Python
        assert "version.killed" in actions
        assert auditlog.verify_chain(events) == []  # Go's hashes verify in Python
        # Python can keep appending after Go.
        auditlog.record(s, cluster.tenant_id, ADMIN, "note.added")
    with scoped_session(cluster.factory, cluster.tenant_id) as s:  # type: ignore[arg-type]
        assert auditlog.verify_chain(auditlog.chain(s, cluster.tenant_id)) == []


def test_the_backend_deploys_through_the_control_plane(cluster: Cluster) -> None:
    from fastapi.testclient import TestClient

    from model_passport.platform.api.app import create_app
    from model_passport.platform.settings import Settings

    c = cluster
    settings = Settings(
        database_url=c.database_url,
        jwt_secret=SECRET,
        master_key=MASTER,
        storage="file://" + tempfile.mkdtemp(),
        controlplane=c.address,
    )
    with TestClient(create_app(settings)) as api:

        def call(who: str, method: str, path: str, expect: int = 200, **body: object) -> object:
            headers = {"Authorization": f"Bearer {c.tokens[who]}", "X-MP-Tenant": "usps"}
            response = api.request(method, f"/api/v1{path}", headers=headers, json=body or None)
            assert response.status_code == expect, response.text
            return response.json()

        for version in ("2.0.0", "2.1.0"):
            vid = c.ids[version]
            call("eng", "POST", f"/versions/{vid}/transition", to="canary")  # deploys to dev
            call("eng", "POST", f"/versions/{vid}/transition", to="verifying")
            call("auditor", "POST", f"/versions/{vid}/approvals", decision="approve")
            call("eng", "POST", f"/versions/{vid}/transition", 409, to="released")
            call("admin", "POST", f"/versions/{vid}/transition", to="released")  # to consumers
        assert c.as_("consumer").resolve("assistant", "consumer")["version"] == "2.1.0"
        live = [d for d in call("eng", "GET", "/deployments") if d["status"] == "active"]
        assert {(d["environment"], d["version"]) for d in live} >= {("consumer", "2.1.0")}

        restored = call("admin", "POST", f"/models/{c.ids['model']}/rollback")
        assert restored["version"] == "2.0.0"  # type: ignore[index]
        assert _state(c, "2.1.0") is State.ROLLED_BACK
        call("eng", "POST", f"/models/{c.ids['model']}/rollback", 403)

        killed = call("auditor", "POST", f"/versions/{c.ids['2.0.0']}/kill", reason="leak found")
        assert killed["state"] == "killed"  # type: ignore[index]
        assert c.as_("consumer").resolve("assistant", "consumer")["found"] is False
        assert call("auditor", "GET", "/audit-log/verify")["intact"] is True  # type: ignore[index]
