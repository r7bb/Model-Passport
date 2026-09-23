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


def _customers(path: Path) -> Path:
    from faker import Faker

    fake = Faker()
    Faker.seed(1)
    rows = ["email,age,plan,churn"] + [
        f"{fake.email()},{20 + i % 50},{'pro' if i % 3 else 'free'},{'yes' if i % 4 else 'no'}"
        for i in range(120)
    ]
    path.write_text("\n".join(rows) + "\n")
    return path


def test_init_with_data_writes_ready_made_stages(tmp_path: Path) -> None:
    import yaml

    from model_passport.core.config import load_config

    data = _customers(tmp_path / "customers.csv")
    result = runner.invoke(app, ["init", str(tmp_path), "--data", str(data), "--label", "churn"])
    assert result.exit_code == 0, result.output
    assert "`churn` is binary-classification" in result.output
    assert "identifiers to drop: email" in result.output
    assert "next: passport run && passport build" in result.output
    config = load_config(tmp_path / "passport.yaml")  # valid against the schema
    assert [s.name for s in config.stages] == ["preprocess", "train", "evaluate"]
    assert config.stages[0].params["input"] == "customers.csv"
    assert config.audit.label_column == "churn"
    assert yaml.safe_load((tmp_path / "passport.yaml").read_text())["privacy"][
        "quasi_identifiers"
    ] == ["age"]
    for stage in ("preprocess", "train", "evaluate"):
        assert f'run("{stage}")' in (tmp_path / f"stages/{stage}.py").read_text()
    assert "rules:" in (tmp_path / "policy.yaml").read_text()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--label", "churn"], "dataset not found"),
        ([], "--label is required"),
        (["--label", "nope"], "is not a column"),
    ],
)
def test_init_with_data_explains_mistakes(tmp_path: Path, args: list[str], message: str) -> None:
    data = _customers(tmp_path / "customers.csv")
    target = "missing.csv" if message == "dataset not found" else str(data)
    result = runner.invoke(app, ["init", str(tmp_path / "p"), "--data", target, *args])
    assert result.exit_code != 0
    assert message in result.output


def test_prepare_without_a_passport_explains_what_to_do(tmp_path: Path) -> None:
    batch = _customers(tmp_path / "batch.csv")
    (tmp_path / "passport.json").write_text('{"pipeline": []}')
    result = runner.invoke(
        app, ["prepare", str(batch), "--passport", str(tmp_path / "passport.json")]
    )
    assert result.exit_code != 0
    assert "no preprocess stage" in result.output
