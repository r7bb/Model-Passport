from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_passport.core import identity
from model_passport.core.builder import BuildError, build_passport, write_passport
from model_passport.core.config import CONFIG_FILENAME
from model_passport.core.verifier import ArtifactStatus, VerificationReport, verify_passport


def _build(project: Path) -> Path:
    out = project / "passport.json"
    write_passport(build_passport(project / CONFIG_FILENAME), out)
    return out


def _verify(project: Path, passport: Path, key: Path | None = None) -> VerificationReport:
    return verify_passport(passport, key or project / ".passport" / "signing_key.pub", project)


def test_build_records_all_artifacts(project: Path) -> None:
    passport = build_passport(project / CONFIG_FILENAME)
    paths = {a.path: a.kind.value for a in passport.artifacts}
    assert paths == {
        "passport.yaml": "config",
        "models/model.pkl": "model",
        "data/train.csv": "dataset",
        "data/test.csv": "dataset",
        "train.py": "script",
    }
    assert passport.model is not None
    assert passport.model.artifact_sha256 == identity.sha256_file(project / "models/model.pkl")
    assert [d.split_role.value for d in passport.datasets] == ["train", "test"]
    assert passport.datasets[0].name == "train"
    assert passport.identity.merkle_root == identity.merkle_root(
        a.sha256 for a in passport.artifacts
    )
    assert passport.identity.signature


def test_fresh_passport_verifies(project: Path) -> None:
    report = _verify(project, _build(project))
    assert report.ok, report
    assert all(a.status is ArtifactStatus.OK for a in report.artifacts)


@pytest.mark.parametrize(
    "relpath", ["models/model.pkl", "data/train.csv", "data/test.csv", "train.py", CONFIG_FILENAME]
)
def test_tampered_artifact_is_named(project: Path, relpath: str) -> None:
    passport = _build(project)
    with (project / relpath).open("ab") as fh:
        fh.write(b"\n# tampered")

    report = _verify(project, passport)
    assert not report.ok
    assert [(c.path, c.status) for c in report.changed] == [(relpath, ArtifactStatus.MODIFIED)]
    # The passport itself is untouched, so its internal integrity still holds.
    assert report.signature_ok and report.merkle_ok


def test_missing_artifact_is_named(project: Path) -> None:
    passport = _build(project)
    (project / "data/test.csv").unlink()
    report = _verify(project, passport)
    assert [(c.path, c.status) for c in report.changed] == [
        ("data/test.csv", ArtifactStatus.MISSING)
    ]


def test_edited_passport_fails_signature(project: Path) -> None:
    passport = _build(project)
    data = json.loads(passport.read_text())
    data["metrics"]["test"]["accuracy"] = 0.99
    passport.write_text(json.dumps(data))

    report = _verify(project, passport)
    assert not report.ok
    assert not report.signature_ok
    assert not report.changed


def test_rewritten_hash_and_merkle_still_fails_signature(project: Path) -> None:
    """An attacker who swaps the model and patches every hash still can't forge the signature."""
    passport = _build(project)
    (project / "models/model.pkl").write_bytes(b"malicious-model")
    new_hash = identity.sha256_file(project / "models/model.pkl")

    data = json.loads(passport.read_text())
    for artifact in data["artifacts"]:
        if artifact["path"] == "models/model.pkl":
            artifact["sha256"] = new_hash
    data["model"]["artifact_sha256"] = new_hash
    data["identity"]["merkle_root"] = identity.merkle_root(a["sha256"] for a in data["artifacts"])
    passport.write_text(json.dumps(data))

    report = _verify(project, passport)
    assert not report.changed and report.merkle_ok
    assert not report.signature_ok
    assert not report.ok


def test_merkle_root_mismatch_detected(project: Path) -> None:
    passport = _build(project)
    data = json.loads(passport.read_text())
    data["identity"]["merkle_root"] = "0" * 64
    passport.write_text(json.dumps(data))
    report = _verify(project, passport)
    assert not report.merkle_ok and not report.ok


def test_wrong_public_key_fails(project: Path, tmp_path: Path) -> None:
    passport = _build(project)
    other_priv, other_pub = tmp_path / "o.pem", tmp_path / "o.pub"
    identity.save_keypair(identity.generate_keypair(), other_priv, other_pub)
    report = _verify(project, passport, key=other_pub)
    assert not report.fingerprint_ok and not report.signature_ok and not report.ok


def test_appended_events_do_not_break_signature(project: Path) -> None:
    passport = _build(project)
    data = json.loads(passport.read_text())
    data["events"].append({"event_type": "drift_check", "payload": {"ks_p": 0.4}})
    passport.write_text(json.dumps(data))
    assert _verify(project, passport).ok


def test_unreadable_passport_reports_error(project: Path) -> None:
    bad = project / "bad.json"
    bad.write_text("{not json")
    report = _verify(project, bad)
    assert report.errors and not report.ok


def test_build_fails_on_missing_model(project: Path) -> None:
    (project / "models/model.pkl").unlink()
    with pytest.raises(BuildError, match="model artifact not found"):
        build_passport(project / CONFIG_FILENAME)


def test_build_fails_without_key(project: Path) -> None:
    (project / ".passport/signing_key.pem").unlink()
    with pytest.raises(BuildError, match="signing key not found"):
        build_passport(project / CONFIG_FILENAME)
