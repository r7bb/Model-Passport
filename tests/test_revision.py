from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from model_passport.cli import app
from model_passport.core.builder import build_passport, write_passport
from model_passport.core.config import CONFIG_FILENAME
from model_passport.core.revision import HISTORY_DIR, archive, compute_revision, load_previous

runner = CliRunner()


def _build(project: Path, **kwargs: object):
    return build_passport(project / CONFIG_FILENAME, **kwargs)


def test_revision_records_dataset_and_artifact_changes(project: Path) -> None:
    first = _build(project)
    (project / "data/train.csv").write_text("age,label\n34,1\n51,0\n40,1\n")
    (project / "notes.txt").write_text("new artifact")
    second = _build(project, previous=first, reason="new rows")

    revision = second.revision
    assert revision is not None
    assert revision.previous_passport_id == first.identity.passport_id
    assert revision.previous_merkle_root == first.identity.merkle_root
    assert revision.sequence == 1
    assert revision.reason == "new rows"
    assert revision.changed_artifacts == ["data/train.csv"]
    status = {c.name: c.status.value for c in revision.dataset_changes}
    assert status == {"train": "changed", "test": "unchanged"}
    train = next(c for c in revision.dataset_changes if c.name == "train")
    assert (train.previous_rows, train.current_rows) == (2, 3)


def test_sequence_increments_and_metric_deltas(project: Path) -> None:
    first = _build(project)
    second = _build(project, previous=first)
    second.metrics["test"]["accuracy"] = 0.95
    third = compute_revision(second, second.model_copy(deep=True), None)
    assert third.sequence == 2
    tweaked = second.model_copy(deep=True)
    tweaked.metrics["test"]["accuracy"] = 0.90
    assert compute_revision(second, tweaked, None).metric_deltas == {
        "test": {"accuracy": pytest.approx(-0.05)}
    }


def test_load_previous_requires_same_model(project: Path, tmp_path: Path) -> None:
    out = tmp_path / "passport.json"
    assert load_previous(out, "demo-model") is None
    write_passport(_build(project), out)
    assert load_previous(out, "demo-model") is not None
    assert load_previous(out, "other-model") is None
    out.write_text("{broken")
    assert load_previous(out, "demo-model") is None


def test_archive_keeps_events(project: Path) -> None:
    out = project / "passport.json"
    write_passport(_build(project), out)
    data = json.loads(out.read_text())
    target = archive(project, out)
    assert target == project / HISTORY_DIR / f"{data['identity']['passport_id']}.json"
    assert json.loads(target.read_text()) == data


def test_cli_build_links_and_archives(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    assert runner.invoke(app, ["build"]).exit_code == 0
    first_id = json.loads((project / "passport.json").read_text())["identity"]["passport_id"]

    result = runner.invoke(app, ["build", "--reason", "retrain"])
    assert result.exit_code == 0, result.output
    assert "supersedes:" in result.output
    second = json.loads((project / "passport.json").read_text())
    assert second["revision"]["previous_passport_id"] == first_id
    assert (project / HISTORY_DIR / f"{first_id}.json").is_file()
    assert (project / "passport.html").is_file()

    result = runner.invoke(app, ["build", "--no-link", "--no-html"])
    assert json.loads((project / "passport.json").read_text())["revision"] is None
    assert runner.invoke(app, ["verify"]).exit_code == 0
