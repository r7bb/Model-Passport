from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import make_frame
from model_passport.cli import app
from model_passport.core import identity
from model_passport.core.verifier import verify_passport
from model_passport.monitoring.monitor import (
    DRIFT_EVENT,
    MonitorError,
    MonitorSettings,
    monitor_batch,
)

runner = CliRunner()


def _key(root: Path) -> identity.Ed25519PrivateKey:
    return identity.load_private_key(root / ".passport/signing_key.pem")


def _batch(root: Path, name: str, **kwargs: object) -> Path:
    path = root / f"{name}.csv"
    make_frame(**kwargs).to_csv(path, index=False)  # type: ignore[arg-type]
    return path


def test_stable_batch_needs_no_retraining(sk_project: Path) -> None:
    batch = _batch(sk_project, "stable", n=800, seed=10)
    result = monitor_batch(sk_project / "passport.json", batch, sk_project, _key(sk_project))
    assert not result.drift.drift_detected
    assert not result.retrain_recommended
    assert result.performance
    assert not result.performance["degraded"]
    assert {f.feature for f in result.drift.features} == {"hours", "group"}  # label excluded


def test_drifted_batch_recommends_retraining_and_appends_signed_event(sk_project: Path) -> None:
    passport = sk_project / "passport.json"
    batch = _batch(sk_project, "drifted", n=800, seed=11, shift=1.0)
    result = monitor_batch(passport, batch, sk_project, _key(sk_project))
    assert set(result.drift.drifted_features) == {"hours", "group"}
    assert result.retrain_recommended

    events = json.loads(passport.read_text())["events"]
    assert len(events) == 1
    assert events[0]["event_type"] == DRIFT_EVENT
    payload = events[0]["payload"]
    assert payload["batch_sha256"] == identity.sha256_file(batch)
    assert payload["reference"] == "train"
    assert verify_passport(passport, sk_project / ".passport/signing_key.pub", sk_project).ok


def test_unlabeled_batch_skips_performance(sk_project: Path) -> None:
    path = sk_project / "unlabeled.csv"
    make_frame(300, 12).drop(columns=["label"]).to_csv(path, index=False)
    result = monitor_batch(
        sk_project / "passport.json", path, sk_project, _key(sk_project), write=False
    )
    assert result.performance is None
    assert not json.loads((sk_project / "passport.json").read_text())["events"]  # write=False


def test_schema_problem_recommends_retraining(sk_project: Path) -> None:
    path = sk_project / "broken.csv"
    make_frame(300, 13).drop(columns=["group"]).to_csv(path, index=False)
    result = monitor_batch(
        sk_project / "passport.json", path, sk_project, _key(sk_project), write=False
    )
    assert result.drift.schema.missing_columns == ["group"]
    assert result.retrain_recommended


def test_changed_reference_is_rejected(sk_project: Path) -> None:
    batch = _batch(sk_project, "b", n=100, seed=14)
    with (sk_project / "data/train.csv").open("a") as fh:
        fh.write("41.0,a,y\n")
    with pytest.raises(MonitorError, match="changed since the passport was built"):
        monitor_batch(sk_project / "passport.json", batch, sk_project, _key(sk_project))


def test_wrong_signing_key_is_rejected(sk_project: Path) -> None:
    batch = _batch(sk_project, "b", n=100, seed=15)
    with pytest.raises(MonitorError, match="signing key"):
        monitor_batch(sk_project / "passport.json", batch, sk_project, identity.generate_keypair())


def test_unknown_reference_dataset(sk_project: Path) -> None:
    batch = _batch(sk_project, "b", n=100, seed=16)
    with pytest.raises(MonitorError, match="no reference dataset"):
        monitor_batch(
            sk_project / "passport.json",
            batch,
            sk_project,
            _key(sk_project),
            MonitorSettings(reference="nope"),
        )


def test_cli_exit_codes(sk_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(sk_project)
    _batch(sk_project, "stable", n=800, seed=17)
    _batch(sk_project, "drifted", n=800, seed=18, shift=1.0)

    result = runner.invoke(app, ["monitor", "drift", "stable.csv"])
    assert result.exit_code == 0, result.output
    assert "no retraining needed" in result.output

    result = runner.invoke(app, ["monitor", "drift", "drifted.csv"])
    assert result.exit_code == 1
    assert "[DRIFT] hours" in result.output
    assert "retraining recommended" in result.output

    assert runner.invoke(app, ["verify"]).exit_code == 0
    assert runner.invoke(app, ["monitor", "drift", "missing.csv"]).exit_code == 2
