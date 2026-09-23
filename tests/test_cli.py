from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from model_passport.cli import app

runner = CliRunner()


def test_init_creates_config_keys_and_gitignore(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path), "--name", "my-model"])
    assert result.exit_code == 0, result.output
    assert "name: my-model" in (tmp_path / "passport.yaml").read_text()
    assert (tmp_path / ".passport/signing_key.pem").is_file()
    assert (tmp_path / ".passport/signing_key.pub").is_file()
    assert ".passport/" in (tmp_path / ".gitignore").read_text().splitlines()
    assert "sha256:" in result.output


def test_init_is_idempotent_and_keeps_key(tmp_path: Path) -> None:
    runner.invoke(app, ["init", str(tmp_path)])
    key_before = (tmp_path / ".passport/signing_key.pem").read_bytes()
    (tmp_path / ".gitignore").write_text("node_modules/\n.passport/\n")

    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0
    assert "kept existing" in result.output
    assert (tmp_path / ".passport/signing_key.pem").read_bytes() == key_before
    assert (tmp_path / ".gitignore").read_text() == "node_modules/\n.passport/\n"


def test_build_and_verify_cli(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    result = runner.invoke(app, ["build"])
    assert result.exit_code == 0, result.output
    assert (project / "passport.json").is_file()

    result = runner.invoke(app, ["verify", "passport.json"])
    assert result.exit_code == 0, result.output
    assert "VERIFIED" in result.output

    (project / "models/model.pkl").write_bytes(b"swapped")
    result = runner.invoke(app, ["verify", "passport.json"])
    assert result.exit_code == 1
    assert "CHANGED  models/model.pkl (model): hash mismatch" in result.output


def test_build_missing_model_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner.invoke(app, ["init", str(tmp_path)])
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["build"])
    assert result.exit_code == 2
    assert "model artifact not found" in result.output
