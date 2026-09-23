from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from model_passport.core import identity
from model_passport.core.builder import build_passport, write_passport
from model_passport.core.config import CONFIG_FILENAME
from model_passport.core.events import append_event
from model_passport.registry.api import create_app
from model_passport.registry.sources import LocalSource

TOKEN = "test-token"


@pytest.fixture
def built(project: Path) -> tuple[dict, str, identity.Ed25519PrivateKey]:
    document = build_passport(project / CONFIG_FILENAME).model_dump(mode="json")
    pem = (project / ".passport/signing_key.pub").read_text()
    return document, pem, identity.load_private_key(project / ".passport/signing_key.pem")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "registry.db", token=""))


def _upload(client: TestClient, document: dict, pem: str, **headers: str):
    return client.post("/passports", json={"passport": document, "public_key_pem": pem},
                       headers=headers)  # fmt: skip


def test_upload_list_get_verify(client: TestClient, built: tuple) -> None:
    document, pem, _ = built
    pid = document["identity"]["passport_id"]
    response = _upload(client, document, pem)
    assert response.status_code == 201, response.text
    assert response.json()["model_name"] == "demo-model"

    assert [p["passport_id"] for p in client.get("/passports").json()] == [pid]
    assert client.get("/passports", params={"model_name": "other"}).json() == []
    assert client.get(f"/passports/{pid}").json() == document
    assert client.get("/models").json()[0]["passports"] == 1

    verification = client.get(f"/passports/{pid}/verify").json()
    assert verification["ok"]
    assert verification["signature_ok"]
    assert not verification["artifacts_checked"]
    assert client.get(f"/passports/{pid}/public-key").json()["public_key_pem"] == pem


def test_rejects_tampered_duplicate_and_wrong_key(client: TestClient, built: tuple) -> None:
    document, pem, _ = built
    tampered = json.loads(json.dumps(document))
    tampered["metrics"]["test"]["accuracy"] = 0.99
    response = _upload(client, tampered, pem)
    assert response.status_code == 422
    assert "signature is invalid" in response.json()["detail"]["problems"]

    other_pem = identity.generate_keypair().public_key()
    from cryptography.hazmat.primitives import serialization

    other = other_pem.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    assert _upload(client, document, other).status_code == 422
    assert _upload(client, document, "not a key").status_code == 422

    assert _upload(client, document, pem).status_code == 201
    assert _upload(client, document, pem).status_code == 409
    assert client.get("/passports/does-not-exist").status_code == 404


def test_token_required_for_writes(tmp_path: Path, built: tuple) -> None:
    document, pem, _ = built
    client = TestClient(create_app(tmp_path / "r.db", token=TOKEN))
    assert _upload(client, document, pem).status_code == 401
    assert _upload(client, document, pem, Authorization="Bearer wrong").status_code == 401
    assert _upload(client, document, pem, Authorization=f"Bearer {TOKEN}").status_code == 201
    assert client.get("/passports").status_code == 200  # reads stay open


def test_events_must_extend_the_chain(client: TestClient, built: tuple) -> None:
    document, pem, key = built
    pid = document["identity"]["passport_id"]
    assert _upload(client, document, pem).status_code == 201

    working = json.loads(json.dumps(document))
    first = append_event(working, "drift_check", {"drift_detected": False}, key)
    assert client.post(f"/passports/{pid}/events", json={"event": first}).json()["events"] == 1
    # Replaying the same event does not extend the chain.
    assert client.post(f"/passports/{pid}/events", json={"event": first}).status_code == 422

    forged = append_event(working, "drift_check", {"drift_detected": True}, key)
    forged["payload"]["drift_detected"] = False
    assert client.post(f"/passports/{pid}/events", json={"event": forged}).status_code == 422
    assert client.get(f"/passports/{pid}/verify").json()["ok"]


def test_lineage_dag_html_jsonld(client: TestClient, project: Path, built: tuple) -> None:
    _, pem, _ = built
    first = build_passport(project / CONFIG_FILENAME)
    second = build_passport(project / CONFIG_FILENAME, previous=first)
    for passport in (first, second):
        assert _upload(client, passport.model_dump(mode="json"), pem).status_code == 201

    sid = str(second.identity.passport_id)
    lineage = client.get(f"/passports/{sid}/lineage").json()
    assert lineage["edges"] == [
        {"from": str(first.identity.passport_id), "to": sid, "relation": "supersedes"}
    ]
    assert len(lineage["nodes"]) == 2
    assert "digraph lineage" in client.get(f"/passports/{sid}/dag").json()["lineage"]
    html = client.get(f"/passports/{sid}/html")
    assert html.status_code == 200
    assert "demo-model" in html.text
    assert "@graph" in client.get(f"/passports/{sid}/jsonld").json()


def test_trusted_keys(tmp_path: Path, project: Path, built: tuple) -> None:
    document, pem, _ = built
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    (trusted / "team.pub").write_text(pem)
    client = TestClient(create_app(tmp_path / "t.db", token="", trusted_keys_dir=str(trusted)))
    _upload(client, document, pem)
    pid = document["identity"]["passport_id"]
    assert client.get(f"/passports/{pid}/verify").json()["key_trusted"] is True


def test_local_source(project: Path) -> None:
    first = build_passport(project / CONFIG_FILENAME)
    write_passport(first, project / ".passport/history" / f"{first.identity.passport_id}.json")
    second = build_passport(project / CONFIG_FILENAME, previous=first)
    write_passport(second, project / "passport.json")
    (project / "not-a-passport.json").write_text("[]")

    source = LocalSource(project)
    entries = source.entries()
    assert [e.passport_id for e in entries] == [
        str(second.identity.passport_id),
        str(first.identity.passport_id),
    ]
    assert "rev 1" in entries[0].label
    assert source.verification(entries[0].passport_id)["ok"]
