from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from model_passport.cli import app
from model_passport.core.builder import build_passport, write_passport
from model_passport.core.config import CONFIG_FILENAME
from model_passport.core.jsonld import to_jsonld
from model_passport.core.schema import (
    Finding,
    LeakageResult,
    Passport,
    PrivacyReport,
    ReidentificationResult,
    Severity,
)
from model_passport.core.verifier import verify_passport
from model_passport.report.dag import lineage_dot, pipeline_dot
from model_passport.report.render import format_threshold, render_html
from model_passport.report.summary import summarize

runner = CliRunner()


@pytest.fixture
def passport(project: Path) -> Passport:
    return build_passport(project / CONFIG_FILENAME)


def _with_privacy(p: Passport, k: int, auc: float, pii: bool) -> Passport:
    findings = [
        Finding(scanner="pii", category="EMAIL", severity=Severity.HIGH, location="train:email",
                masked_examples=["j***@e***.com"], details={"direct_identifier": True})
    ] if pii else []  # fmt: skip
    p = p.model_copy(deep=True)
    p.privacy_report = PrivacyReport(
        datasets_scanned=["train"],
        data_findings=findings,
        reidentification=[
            ReidentificationResult(
                dataset="train",
                rows=10,
                quasi_identifiers=["age"],
                k_anonymity=k,
                unique_fraction=0.0,
            )
        ],  # fmt: skip
        leakage=LeakageResult(mia_auc=auc, tpr_at_low_fpr=0.0, members=5, nonmembers=5),
    )
    return p


def test_summary_plain_language(passport: Passport) -> None:
    good = summarize(_with_privacy(passport, k=8, auc=0.51, pii=False))
    assert not good.concerns
    assert any("8-anonymity" in p for p in good.points)
    assert any("coin flip" in p for p in good.points)

    bad = summarize(_with_privacy(passport, k=1, auc=0.9, pii=True))
    assert any("identifiers" in c for c in bad.concerns)
    assert any("singled out" in c for c in bad.concerns)
    assert any("memorizes" in c for c in bad.concerns)


def test_html_escapes_and_contains_sections(passport: Passport) -> None:
    hostile = passport.model_copy(deep=True)
    hostile.declared.intended_use = "<script>alert(1)</script>"
    html = render_html(_with_privacy(hostile, k=8, auc=0.5, pii=True))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    for section in ('id="summary"', 'id="model"', 'id="privacy"', 'id="identity"'):
        assert section in html
    assert "j***@e***.com" in html  # masked examples only


def test_html_with_verification(project: Path, passport: Passport) -> None:
    out = project / "passport.json"
    write_passport(passport, out)
    report = verify_passport(out, project / ".passport/signing_key.pub", project)
    html = render_html(passport, report)
    assert 'id="verification"' in html and "verified" in html


def test_format_threshold() -> None:
    assert format_threshold({"warn": 0.55, "fail": 0.6}) == "warn 0.55 · fail 0.6"
    assert format_threshold(5) == "5"


def test_jsonld_uses_prov_and_content_addresses(passport: Passport) -> None:
    doc = to_jsonld(passport)
    assert doc["@context"]["prov"] == "http://www.w3.org/ns/prov#"
    nodes = {n["@id"]: n for n in doc["@graph"]}
    for artifact in passport.artifacts:
        assert f"urn:sha256:{artifact.sha256}" in nodes
    root = nodes[f"urn:uuid:{passport.identity.passport_id}"]
    assert "prov:Entity" in root["@type"]
    assert root["mp:merkleRoot"] == passport.identity.merkle_root
    json.dumps(doc)  # serializable


def test_dot_quotes_names(passport: Passport) -> None:
    p = passport.model_copy(deep=True)
    p.identity.model_name = 'weird "name"'
    assert '\\"name\\"' in lineage_dot(p)
    assert pipeline_dot(p).startswith("digraph pipeline")


def test_cli_report_and_export(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    assert runner.invoke(app, ["build", "--no-html"]).exit_code == 0
    result = runner.invoke(app, ["report", "passport.json", "--root", "."])
    assert result.exit_code == 0 and (project / "passport.html").is_file()
    result = runner.invoke(app, ["export", "passport.json"])
    assert result.exit_code == 0
    assert "@graph" in json.loads((project / "passport.jsonld").read_text())
    assert runner.invoke(app, ["report", "missing.json"]).exit_code == 2
