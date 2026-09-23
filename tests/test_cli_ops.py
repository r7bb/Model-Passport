"""CLI commands not covered elsewhere: scan, audit, push."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from faker import Faker
from typer.testing import CliRunner

from model_passport.cli import app, ops
from model_passport.registry.client import RegistryError

runner = CliRunner()


def test_scan_data_and_fail_on(tmp_path: Path) -> None:
    fake = Faker()
    Faker.seed(3)
    path = tmp_path / "people.csv"
    path.write_text("email,age\n" + "\n".join(f"{fake.email()},{30 + i}" for i in range(20)))
    out = tmp_path / "findings.json"
    result = runner.invoke(app, ["scan", "data", str(path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "EMAIL" in result.output
    assert any(f["category"] == "EMAIL" for f in json.loads(out.read_text()))
    assert runner.invoke(app, ["scan", "data", str(path), "--fail-on", "high"]).exit_code == 1
    assert runner.invoke(app, ["scan", "data", str(tmp_path / "nope.csv")]).exit_code == 2


def test_scan_secrets_directory(tmp_path: Path) -> None:
    (tmp_path / "clean.py").write_text("x = 1\n")
    assert runner.invoke(app, ["scan", "secrets", str(tmp_path)]).exit_code == 0
    (tmp_path / "leak.env").write_text("-----BEGIN RSA PRIVATE KEY-----\n")
    result = runner.invoke(app, ["scan", "secrets", str(tmp_path)])
    assert result.exit_code == 1
    assert "PRIVATE_KEY" in result.output


def test_audit_model_cli(sk_project: Path) -> None:
    out = sk_project / "audit.json"
    result = runner.invoke(
        app,
        ["audit", "model", str(sk_project / "models/model.pkl"),
         "--members", str(sk_project / "data/train.csv"),
         "--nonmembers", str(sk_project / "data/test.csv"),
         "--label", "label", "--string-cols", "group", "--out", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "membership inference AUC" in result.output
    assert 0.0 <= json.loads(out.read_text())["leakage"]["mia_auc"] <= 1.0


def test_push_success_and_failure(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    assert runner.invoke(app, ["build", "--no-html"]).exit_code == 0
    sent: dict = {}

    def fake_upload(self, passport: dict, pem: str) -> dict:
        sent.update(passport=passport, pem=pem)
        ident = passport["identity"]
        return {"model_name": ident["model_name"], "version": ident["version"],
                "passport_id": ident["passport_id"], "verdict": None}  # fmt: skip

    monkeypatch.setattr(ops.RegistryClient, "upload", fake_upload)
    result = runner.invoke(app, ["push", "--registry", "http://registry.test"])
    assert result.exit_code == 0, result.output
    assert "uploaded demo-model" in result.output
    assert sent["pem"].startswith("-----BEGIN PUBLIC KEY-----")

    def refuse(self, passport: dict, pem: str) -> dict:
        raise RegistryError("HTTP 422: passport failed verification")

    monkeypatch.setattr(ops.RegistryClient, "upload", refuse)
    result = runner.invoke(app, ["push", "--registry", "http://registry.test"])
    assert result.exit_code == 1
    assert "422" in result.output
