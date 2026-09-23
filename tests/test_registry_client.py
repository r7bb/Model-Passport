"""The stdlib client against a real registry served by uvicorn on a local port."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn
from typer.testing import CliRunner

from model_passport.cli import app
from model_passport.core import identity
from model_passport.core.events import append_event
from model_passport.registry.api import create_app
from model_passport.registry.client import RegistryClient, RegistryError

TOKEN = "s3cret-for-tests"
runner = CliRunner()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def registry_url(tmp_path: Path) -> Iterator[str]:
    port = _free_port()
    config = uvicorn.Config(
        create_app(tmp_path / "registry.db", token=TOKEN), host="127.0.0.1", port=port,
        log_level="warning",
    )  # fmt: skip
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("registry did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_push_list_verify_and_events(
    project: Path, registry_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(project)
    assert runner.invoke(app, ["build", "--no-html"]).exit_code == 0
    denied = runner.invoke(app, ["push", "--registry", registry_url, "--token", "wrong"])
    assert denied.exit_code == 1
    assert "401" in denied.output
    result = runner.invoke(app, ["push", "--registry", registry_url, "--token", TOKEN])
    assert result.exit_code == 0, result.output

    client = RegistryClient(registry_url, TOKEN)
    document = json.loads((project / "passport.json").read_text())
    pid = document["identity"]["passport_id"]
    assert [p["passport_id"] for p in client.search(model_name="demo-model")] == [pid]
    assert client.get(pid)["identity"]["merkle_root"] == document["identity"]["merkle_root"]
    assert client.verify(pid)["ok"]
    assert client.lineage(pid)["root"] == pid
    assert "digraph pipeline" in client.dag(pid)["pipeline"]

    key = identity.load_private_key(project / ".passport/signing_key.pem")
    event = append_event(document, "drift_check", {"drift_detected": False}, key)
    assert client.append_event(pid, event)["events"] == 1
    with pytest.raises(RegistryError, match="422"):
        client.append_event(pid, event)


def test_unreachable_registry() -> None:
    client = RegistryClient(f"http://127.0.0.1:{_free_port()}", timeout=2)
    with pytest.raises(RegistryError, match="cannot reach registry"):
        client.search()
