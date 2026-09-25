"""Language model projects: scaffolding, versioning, and the remediation loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from model_passport.cli import app
from model_passport.core.verifier import verify_passport
from model_passport.llm import synthetic
from model_passport.llm.entities import Record, Span, write_corpus
from model_passport.llm.remediate import (
    RemediationSettings,
    bump_minor,
    gate_passed,
    remediate,
    set_version,
    translate,
)

runner = CliRunner()


def test_versions_bump_and_rewrite_in_place(tmp_path: Path) -> None:
    assert bump_minor("1.0.0") == "1.1.0"
    assert bump_minor("2.9") == "2.10.0"
    assert bump_minor("v1") == "v1.1"
    config = tmp_path / "passport.yaml"
    config.write_text("# keep me\nproject:\n  name: m\n  version: 1.0.0\nbuild:\n  version: 7\n")
    set_version(config, "1.1.0")
    assert config.read_text() == (
        "# keep me\nproject:\n  name: m\n  version: 1.1.0\nbuild:\n  version: 7\n"
    )


def test_translate_follows_entities_whose_offsets_moved() -> None:
    raw = [Record("a", "Mail ann@x.io or bob@y.io", [Span(5, 13, "EMAIL"), Span(17, 25, "EMAIL")])]
    # An earlier round replaced the first email with a longer surrogate.
    train = [Record("a", "Mail joanna@zz.org or bob@y.io",
                    [Span(5, 18, "EMAIL"), Span(22, 30, "EMAIL")])]  # fmt: skip
    wanted = {"a": {(17, 25)}}
    assert translate(wanted, raw, train) == {"a": {(22, 30)}}
    both = {"a": {(5, 13), (17, 25)}}
    assert translate(both, raw, train, only_original=True) == {"a": {(22, 30)}}


def test_init_llm_writes_a_runnable_project(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(synthetic.corpus(20, seed=0), corpus)
    project = tmp_path / "p"
    result = runner.invoke(app, ["init", str(project), "--llm", "--corpus", str(corpus)])
    assert result.exit_code == 0, result.output
    for name in ("passport.yaml", "policy.yaml", "stages/finetune.py", "stages/audit.py",
                 "data/raw.jsonl", "data/train.jsonl"):  # fmt: skip
        assert (project / name).is_file(), name
    config = (project / "passport.yaml").read_text()
    assert "entity_audit: reports/entity_audit.json" in config
    assert "epochs: 15" in config  # demo settings for a model trained from scratch
    missing = runner.invoke(app, ["init", str(tmp_path / "q"), "--llm"])
    assert missing.exit_code != 0
    assert "--corpus is required" in missing.output


def test_remediation_loop_versions_until_the_gate_passes(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(synthetic.corpus(160, seed=2), corpus)
    project = tmp_path / "p"
    result = runner.invoke(app, ["init", str(project), "--llm", "--corpus", str(corpus)])
    assert result.exit_code == 0, result.output

    rounds = remediate(project / "passport.yaml", RemediationSettings(max_rounds=3))
    first, last = rounds[0], rounds[-1]
    assert first.verdict == "fail"
    assert first.blocking > 0
    assert gate_passed(rounds)
    assert last.blocking == 0
    assert [r.version for r in rounds] == [f"1.{i}.0" for i in range(len(rounds))]
    history = sorted(p.name for p in (project / "data/history").iterdir())
    assert history == [f"train.v1.{i}.0.jsonl" for i in range(len(rounds) - 1)]
    passport = json.loads((project / "passport.json").read_text())
    assert passport["revision"]["sequence"] == len(rounds) - 1
    assert passport["revision"]["reason"] == "Remediated memorized entities and retrained"
    archived = list((project / ".passport/history").iterdir())
    assert len(archived) == len(rounds) - 1
    report = verify_passport(
        project / "passport.json", project / ".passport/signing_key.pub", project
    )
    assert report.ok
    raw = (project / "data/raw.jsonl").read_text()
    train = (project / "data/train.jsonl").read_text()
    assert raw != train  # risky values were replaced in the training copy only
