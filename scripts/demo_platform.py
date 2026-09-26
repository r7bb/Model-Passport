"""Run the whole platform locally with a realistic, fully synthetic organization.

Starts PostgreSQL (pgserver), the Go control plane (when Go is installed), and the backend on
``--port``, then walks the real product flow over HTTP: train, audit, remediate, canary test,
approve, release. Every name, email, and record comes from Faker; nothing is real. Leave it
running and start the web app against it::

    python scripts/demo_platform.py            # backend on http://localhost:8080
    cd web && MP_API_URL=http://localhost:8080 npm run dev

Sign in as any of the printed accounts (password: ``demo password 2026``).
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pgserver
import uvicorn
from faker import Faker

from model_passport.llm import synthetic
from model_passport.llm.entities import write_corpus
from model_passport.platform import services
from model_passport.platform.api.app import create_app
from model_passport.platform.db import (
    ALL_TENANTS,
    make_engine,
    migrate,
    scoped_session,
    session_factory,
)
from model_passport.platform.settings import Settings
from model_passport.platform.storage import open_store
from model_passport.platform.worker import WorkerContext, drain

REPO = Path(__file__).resolve().parents[1]
PASSWORD = "demo password 2026"  # noqa: S105 - a throwaway local demo account
ROLES = {
    "eng": "ml_engineer",
    "auditor": "compliance_auditor",
    "tester": "canary_tester",
    "reviewer": "external_reviewer",
    "consumer": "end_consumer",
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for(port: int, what: str) -> None:
    for _ in range(150):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    sys.exit(f"{what} did not start on port {port}")


def _postgres(root: Path) -> tuple[str, str]:
    server = pgserver.get_server(str(root / "pg"))
    server.psql("CREATE ROLE mp_app LOGIN PASSWORD 'pw'; CREATE DATABASE mp OWNER mp_app;")
    host = server.get_uri().split("host=")[1]
    return (
        f"postgresql+psycopg://mp_app:pw@/mp?host={host}",
        f"postgresql://mp_app:pw@/mp?host={host}",
    )


def _controlplane(
    root: Path, dsn: str, secret: str
) -> tuple[str | None, subprocess.Popen[str] | None]:
    go = shutil.which("go") or str(Path.home() / ".local/bin/go")
    if not Path(go).exists():
        print("Go not found: running without the control plane (no deployments page).")
        return None, None
    binary = root / "mp-controlplane"
    subprocess.run(
        [go, "build", "-o", str(binary), "./cmd/mp-controlplane"],
        cwd=REPO / "controlplane",
        check=True,
    )
    port = _free_port()
    env = os.environ | {
        "MP_DATABASE_URL": dsn,
        "MP_JWT_SECRET": secret,
        "MP_CONTROLPLANE_ADDR": f"127.0.0.1:{port}",
    }
    process = subprocess.Popen([str(binary)], env=env, text=True)
    _wait_for(port, "control plane")
    return f"127.0.0.1:{port}", process


class Client:
    def __init__(self, base: str) -> None:
        self.http = httpx.Client(base_url=f"{base}/api/v1", timeout=120)

    def login(self, email: str, org: str | None = None) -> dict[str, str]:
        token = self.call("POST", "/auth/login", {}, json={"email": email, "password": PASSWORD})
        headers = {"Authorization": f"Bearer {token['access_token']}"}
        return headers | ({"X-MP-Tenant": org} if org else {})

    def call(self, method: str, path: str, headers: dict[str, str], **kwargs: Any) -> Any:
        response = self.http.request(method, path, headers=headers, **kwargs)
        if response.status_code >= 400:
            sys.exit(f"{method} {path} failed: {response.status_code} {response.text}")
        return response.json() if response.content else None


def _people(api: Client, root: dict[str, str], fake: Faker) -> dict[str, dict[str, str]]:
    """Two organizations with synthetic staff; returns sessions for the main one."""
    accounts = []
    for slug, name in (("northwind", "Northwind Health"), ("contoso", "Contoso Logistics")):
        api.call(
            "POST", "/platform/tenants", root, json={"slug": slug, "name": name, "plan": "team"}
        )
        admin = {"email": f"admin@{slug}.example", "name": fake.name(), "password": PASSWORD}
        api.call(
            "POST", f"/platform/tenants/{slug}/admins", root, json=admin | {"role": "org_admin"}
        )
        accounts.append((slug, admin["email"], "org_admin"))
    admin = api.login("admin@northwind.example", "northwind")
    sessions = {"admin": admin}
    for who, role in ROLES.items():
        person = fake.name()
        email = f"{person.split()[0].lower()}.{who}@northwind.example"
        body = {"email": email, "name": person, "role": role, "password": PASSWORD}
        api.call("POST", "/members", admin, json=body)
        sessions[who] = api.login(email, "northwind")
        accounts.append(("northwind", email, role))
    print("\nAccounts (password: demo password 2026)")
    print("  super admin              root@platform.example")
    for slug, email, role in accounts:
        print(f"  {slug:10} {role:18} {email}")
    return sessions


def _status(v: dict[str, Any]) -> str:
    return f"  {v['version']}: {v['state']}, critical={v['critical']} high={v['high']}"


def _flow(api: Client, worker: WorkerContext, who: dict[str, dict[str, str]], work: Path) -> None:
    eng, auditor, admin, tester = who["eng"], who["auditor"], who["admin"], who["tester"]
    corpus = work / "support-tickets.jsonl"
    write_corpus(synthetic.corpus(300, seed=7), corpus)
    provenance = {
        "name": "support-tickets-2025",
        "source": "helpdesk export (synthetic)",
        "license": "internal",
        "consent": "obtained",
    }
    files = {"file": (corpus.name, corpus.read_bytes())}
    data = api.call("POST", "/datasets", eng, files=files, data=provenance)
    claims = work / "claims-notes.jsonl"
    write_corpus(synthetic.corpus(200, seed=11), claims)
    other = {
        "name": "claims-notes",
        "source": "vendor share (synthetic)",
        "license": "unknown",
        "consent": "unknown",
    }
    claims_data = api.call(
        "POST", "/datasets", eng, files={"file": (claims.name, claims.read_bytes())}, data=other
    )

    print("\nPhase 1 · detect: training and auditing support-assistant 1.0.0 …")
    model = api.call(
        "POST",
        "/models",
        eng,
        json={
            "name": "support-assistant",
            "access": "open-weight",
            "base": "tiny",
            "description": "Answers patient support tickets",
        },
    )
    request = {
        "version": "1.0.0",
        "dataset_id": data["id"],
        "reference_dataset_id": data["id"],
        "train": True,
    }
    current = api.call("POST", f"/models/{model['id']}/versions", eng, json=request)
    drain(worker)
    current = api.call("GET", f"/versions/{current['id']}", eng)
    print(_status(current))

    print("Phase 2 · remediate: sanitize, retrain, re-audit …")
    for _ in range(4):
        if current["state"] == "clean":
            break
        api.call("POST", f"/versions/{current['id']}/remediate", eng)
        drain(worker)
        current = api.call("GET", f"/models/{model['id']}", eng)["versions"][-1]
        print(_status(current))

    if current["state"] == "clean":
        print("Phase 3 · verify & deploy: canary, approval, release …")
        vid = current["id"]
        api.call("POST", f"/versions/{vid}/transition", eng, json={"to": "canary"})
        report = {
            "prompt_count": 120,
            "leaks_found": 0,
            "summary": "Prefix completion, role-play, and repeat-after-me prompts; nothing leaked.",
        }
        api.call("POST", f"/versions/{vid}/test-reports", tester, json=report)
        api.call("POST", f"/versions/{vid}/transition", eng, json={"to": "verifying"})
        api.call(
            "POST",
            f"/versions/{vid}/approvals",
            auditor,
            json={"decision": "approve", "comment": "Re-audit clean; canary found no leaks."},
        )
        api.call("POST", f"/versions/{vid}/transition", admin, json={"to": "released"})

    print("A second model, left with open findings …")
    second = api.call(
        "POST",
        "/models",
        eng,
        json={
            "name": "claims-summarizer",
            "access": "open-weight",
            "base": "tiny",
            "description": "Summarizes insurance claim notes",
        },
    )
    request = {
        "version": "1.0.0",
        "dataset_id": claims_data["id"],
        "reference_dataset_id": claims_data["id"],
        "train": True,
    }
    api.call("POST", f"/models/{second['id']}/versions", eng, json=request)
    drain(worker)
    api.call("POST", f"/models/{model['id']}/reports", who["reviewer"])
    drain(worker)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--dir", type=Path, default=None, help="where to keep the demo's data")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    root = args.dir or Path(tempfile.mkdtemp(prefix="mp-demo-"))
    root.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_urlsafe(48)
    url, dsn = _postgres(root)
    migrate(make_engine(url))
    address, plane = _controlplane(root, dsn, secret)
    settings = Settings(
        database_url=url,
        jwt_secret=secret,
        master_key=os.urandom(32),
        storage=f"file://{root}/objects",
        work_dir=root / "work",
        controlplane=address,
    )
    factory = session_factory(make_engine(url))
    with scoped_session(factory, ALL_TENANTS) as session:
        services.create_user(
            session, "root@platform.example", PASSWORD, "Platform Root", super_admin=True
        )

    server = uvicorn.Server(
        uvicorn.Config(create_app(settings), port=args.port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    _wait_for(args.port, "backend")

    worker = WorkerContext(
        factory=factory,
        store=open_store(settings.storage),
        master_key=settings.master_key,
        name="demo-worker",
    )
    api = Client(f"http://127.0.0.1:{args.port}")
    fake = Faker()
    Faker.seed(2026)
    who = _people(api, api.login("root@platform.example"), fake)
    _flow(api, worker, who, root)

    print(f"\nBackend ready on http://localhost:{args.port} (data in {root}). Ctrl-C to stop.")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.is_set():
            drain(worker)  # serve jobs started from the web app
            stop.wait(2)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        if plane:
            plane.terminate()


if __name__ == "__main__":
    main()
