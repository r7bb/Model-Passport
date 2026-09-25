"""Entity-level leakage audits: detection, references, methods, statistics, and remediation.

Models are tiny GPT-2 networks trained offline on synthetic records (Faker), so every test runs
without network access and no real personal data is involved.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from model_passport.cli import app
from model_passport.core import identity
from model_passport.core.schema import EntityAuditResult, Severity
from model_passport.llm import entities, synthetic
from model_passport.llm.audit import AuditSettings, audit
from model_passport.llm.backends import ModelError, Scored
from model_passport.llm.evidence import gather
from model_passport.llm.methods import Parts, Sample, min_k, reference_set
from model_passport.llm.references import ReferenceSampler, shape
from model_passport.llm.risk import attack_strength, benjamini_hochberg, entity_risk, severity
from model_passport.llm.sanitize import risky_spans, sanitize

runner = CliRunner()
AUDIT = AuditSettings(max_entities=240, seed=0)


# --- Entities ----------------------------------------------------------------------------------


def test_detect_finds_structured_pii_without_overlaps() -> None:
    text = (
        "Mail jane.doe@example.com or call (555) 123-4567; SSN 123-45-6789, "
        "card 4111 1111 1111 1111, from 10.0.0.12 on 2024-01-31."
    )
    found = {(s.entity_type, text[s.start : s.end]) for s in entities.detect(text)}
    assert found == {
        ("EMAIL", "jane.doe@example.com"),
        ("PHONENUMBER", "(555) 123-4567"),
        ("SSN", "123-45-6789"),
        ("CREDITCARDNUMBER", "4111 1111 1111 1111"),
        ("IPV4", "10.0.0.12"),
        ("DATE", "2024-01-31"),
    }


def test_corpus_formats(tmp_path: Path) -> None:
    ours = {"id": "a", "text": "Hi Ana", "entities": [{"start": 3, "end": 6, "type": "firstname"}]}
    ai4privacy = {"source_text": "Mail x@y.io", "privacy_mask": [
        {"value": "x@y.io", "start": 5, "end": 11, "label": "EMAIL"}]}  # fmt: skip
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps(ours) + "\n" + json.dumps(ai4privacy) + "\n")
    records = list(entities.load_corpus(path))
    assert [(r.id, r.value(r.entities[0]), r.entities[0].entity_type) for r in records] == [
        ("a", "Ana", "FIRSTNAME"),
        ("1", "x@y.io", "EMAIL"),
    ]
    text = tmp_path / "c.txt"
    text.write_text("reach me at ana@example.org\n")
    assert next(entities.load_corpus(text)).entities[0].entity_type == "EMAIL"


def test_impact_levels_follow_sensitivity() -> None:
    assert entities.impact("SSN") == 1.0
    assert entities.impact("email") == 0.8
    assert entities.impact("CITY") == 0.6
    assert entities.impact("EYECOLOR") == 0.4


# --- References --------------------------------------------------------------------------------


def test_references_are_same_type_new_values_with_matching_shape() -> None:
    records = synthetic.corpus(50, seed=3)
    seen = {r.value(s) for r in records for s in r.entities}
    sampler = ReferenceSampler(records, source="synthetic", seed=1)
    refs = sampler.sample("SSN", "123-45-6789", 5)
    assert len(refs) == 5
    assert all(shape(r) == "999-99-9999" for r in refs)
    assert not seen & set(refs)  # synthetic references were never trained on
    emails = sampler.sample("EMAIL", "ana@example.org", 5)
    assert all("@" in e for e in emails)
    unknown = sampler.sample("TICKETCODE", "AB-1234", 3)
    assert all(shape(u) == "AA-9999" for u in unknown)


def test_corpus_references_come_from_other_training_values() -> None:
    records = synthetic.corpus(80, seed=4)
    pool = {r.value(s) for r in records for s in r.entities if s.entity_type == "EMAIL"}
    sampler = ReferenceSampler(records, source="corpus", seed=0)
    candidate = next(iter(pool))
    refs = sampler.sample("EMAIL", candidate, 4)
    assert set(refs) <= pool
    assert candidate not in refs


# --- Methods and statistics --------------------------------------------------------------------


def _scored(offsets: list[tuple[int, int]], logprobs: Sequence[float]) -> Scored:
    return Scored(offsets, np.asarray(logprobs, dtype=float))


def test_parts_split_value_and_continuation() -> None:
    sample = Sample("Hi ", "Ana", " there")
    offsets = [(0, 2), (2, 3), (3, 6), (6, 12)]
    parts = Parts.of(_scored(offsets, [np.nan, -1.0, -2.0, -3.0]), sample.span)
    assert parts.all.tolist() == [-1.0, -2.0, -3.0]
    assert parts.value.tolist() == [-2.0]
    assert parts.suffix.tolist() == [-3.0]
    context = Parts.of(_scored(offsets, [np.nan, -1.0, -2.0, -3.0]), sample.span, text_start=3)
    assert context.all.tolist() == [-2.0, -3.0]


def test_reference_set_is_the_log_likelihood_ratio_against_references() -> None:
    def parts(value_lp: float) -> Parts:
        return Parts(np.array([value_lp]), np.array([value_lp]), np.array([]))

    score = reference_set(parts(-1.0), [parts(-3.0), parts(-5.0)], suffix=False)
    expected = -1.0 - np.log(np.mean(np.exp([-3.0, -5.0])))
    assert score == pytest.approx(expected)
    assert (
        min_k(Parts(np.array([-1.0, -9.0, -2.0, -8.0, -3.0]), np.array([]), np.array([])), 0.4)
        == -8.5
    )


def test_benjamini_hochberg_matches_the_definition() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.5])
    assert benjamini_hochberg(p).tolist() == pytest.approx([0.04, 0.0533333, 0.0533333, 0.5])


def test_risk_is_likelihood_times_impact_in_cvss_bands() -> None:
    assert entity_risk(0.001, 0.01, 1.0, 0.05) == (0.999, 9.99)
    assert severity(9.99) is Severity.CRITICAL
    assert severity(7.0) is Severity.HIGH
    assert severity(4.0) is Severity.MEDIUM
    assert severity(0.5) is Severity.LOW
    assert severity(0.0) is Severity.INFO
    # Without significance an entity is capped at Low, however extreme it looks.
    assert entity_risk(0.001, 0.2, 1.0, 0.05)[1] == 3.9


def test_attack_strength_reports_auc_and_low_fpr_rates() -> None:
    auc, rates = attack_strength([5, 6, 7, 8], [1, 2, 3, 4])
    assert auc == 1.0
    assert rates == {"0.001": 1.0, "0.01": 1.0, "0.05": 1.0}
    chance, _ = attack_strength([1, 2, 3, 4], [1, 2, 3, 4])
    assert chance == 0.5


def test_evidence_picks_the_informative_method_by_cross_fitting() -> None:
    rng = np.random.default_rng(0)
    n = 400
    member = {"weak": rng.normal(0, 1, n), "strong": rng.normal(2.5, 1, n)}
    control = {"weak": rng.normal(0, 1, n), "strong": rng.normal(0, 1, n)}
    evidence = gather(["EMAIL"] * n, member, control, "auto", seed=0)
    assert evidence.primary == "strong"
    assert evidence.p[np.argmax(evidence.z)] < 1e-3
    fixed = gather(["EMAIL"] * n, member, control, "weak", seed=0)
    assert fixed.primary == "weak"


# --- Audits on a trained model -------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained() -> dict[str, Any]:
    pytest.importorskip("torch")
    from model_passport.llm import training

    records = synthetic.corpus(300, seed=1)
    filler = [r.text for r in synthetic.corpus(1200, seed=99)]
    settings = training.TrainSettings(epochs=15, learning_rate=2e-3, batch_size=32)
    model = training.tiny_model([r.text for r in records] + filler, seed=0)
    training.finetune(model, [r.text for r in records], settings)
    return {"records": records, "model": model, "filler": filler, "settings": settings}


def test_audit_detects_memorized_entities(trained: dict[str, Any]) -> None:
    result = audit(trained["model"], trained["records"], AUDIT)
    assert result.access == "logprobs"
    assert result.auc is not None
    assert result.auc > 0.85
    assert result.tpr_at_fpr["0.01"] > 0.1  # far above the 1% a guessing attacker gets
    assert result.severity_counts["critical"] > 0
    assert {m.method for m in result.metrics} >= {"loss", "reference_set", "reference_set_suffix"}
    worst = result.findings[0]
    assert worst.severity is Severity.CRITICAL
    assert worst.confirmed is True  # independently re-tested against 100 fresh alternatives
    assert worst.exposure is not None
    assert worst.exposure > 5  # ranked in the top ~3% of 101 candidates
    assert worst.impact == 1.0  # only fraud-enabling values (cards, SSNs, IBANs) reach critical
    raw = {r.value(s) for r in trained["records"] for s in r.entities}
    assert not raw & {f.masked_value for f in result.findings}  # never stores raw values
    assert all(f.fingerprint is None for f in result.findings)  # no key, no reversible hash


def test_fingerprints_are_keyed(trained: dict[str, Any]) -> None:
    settings = AuditSettings(max_entities=20, seed=0, fingerprint_key=b"tenant-secret")
    result = audit(trained["model"], trained["records"], settings)
    assert all(f.fingerprint and len(f.fingerprint) == 64 for f in result.findings)


def test_sanitize_and_retrain_removes_the_risk(trained: dict[str, Any]) -> None:
    from model_passport.llm import training

    records = trained["records"]
    before = audit(trained["model"], records, AUDIT)
    targets = risky_spans(before, records, Severity.LOW, always=["SSN", "CREDITCARDNUMBER"])
    cleaned, report = sanitize(records, targets)
    assert sum(report.replaced.values()) > 100
    model = training.tiny_model([r.text for r in cleaned] + trained["filler"], seed=0)
    training.finetune(model, [r.text for r in cleaned], trained["settings"])
    after = audit(model, records, AUDIT)
    assert after.auc is not None
    assert after.auc < 0.62
    assert after.severity_counts["critical"] == 0
    assert after.severity_counts["high"] == 0
    # Anything the screen flagged near chance level failed independent confirmation.
    assert all(f.confirmed is not True for f in after.findings if f.risk <= 3.9)


def test_saved_models_reload_and_hash_as_directories(
    trained: dict[str, Any], tmp_path: Path
) -> None:
    from model_passport.llm import training
    from model_passport.llm.backends import load_model

    directory = training.save(trained["model"], tmp_path / "v1")
    assert (directory / "model.safetensors").is_file()
    digest = identity.sha256_path(directory)
    assert digest == identity.sha256_path(directory)
    reloaded = load_model(str(directory), device="cpu")
    assert reloaded.score(["hello"])[0].logprobs.size >= 1  # type: ignore[union-attr]
    (directory / "config.json").write_text((directory / "config.json").read_text() + " ")
    assert identity.sha256_path(directory) != digest


# --- Generation-only models ----------------------------------------------------------------------


class Parrot:
    """A generation-only model that regurgitates the values it memorized."""

    name = "parrot"

    def __init__(self, memorized: dict[str, str]) -> None:
        self.memorized = memorized

    def complete(self, prompt: str, *, n: int = 1, max_tokens: int = 32,
                 temperature: float = 1.0) -> list[str]:  # fmt: skip
        for prefix, value in self.memorized.items():
            if prompt.endswith(prefix):
                return [value + " and more"] * n
        return ["I cannot recall that."] * n


def test_probing_flags_values_a_model_reproduces() -> None:
    records = synthetic.corpus(60, seed=5)
    leaked, leaked_ids = {}, set()
    for record in records[:20]:
        span = record.entities[0]
        prefix = record.text[: span.start]
        if prefix not in leaked and len(leaked) < 10:  # templates can share a prefix
            leaked[prefix] = record.value(span)
            leaked_ids.add(record.id)
    shared = {r.id for r in records if r.entities and r.text[: r.entities[0].start] in leaked}
    result = audit(Parrot(leaked), records, AuditSettings(probe_samples=4, seed=0))
    assert result.access == "generation"
    assert result.primary_method == "extraction"
    flagged = {(f.record, f.extraction_rate) for f in result.findings if f.q_value <= 0.05}
    assert leaked_ids <= {r for r, _ in flagged} <= shared
    assert all(rate == 1.0 for _, rate in flagged)
    # Only 10 of the entities leak, so AUC sits just above 0.5; the true-positive rate at a low
    # false-positive rate shows the attack finds exactly those, with no false alarms.
    rounding = 1e-3  # rates are stored to four decimals
    assert result.tpr_at_fpr["0.001"] >= len(leaked_ids) / result.entities_audited - rounding


# --- API backends ------------------------------------------------------------------------------


class _Response:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.data


def test_openai_compatible_scoring_uses_echoed_prompt_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from model_passport.llm.backends import OpenAICompatibleModel

    def fake_post(url: str, **kwargs: Any) -> _Response:
        assert url.endswith("/completions")
        assert kwargs["json"]["echo"] is True
        return _Response({"choices": [{"logprobs": {
            "tokens": ["Hi", " Ana", "!"], "text_offset": [0, 2, 6],
            "token_logprobs": [None, -1.5, -7.0]}}]})  # fmt: skip

    monkeypatch.setattr(httpx, "post", fake_post)
    scored = OpenAICompatibleModel("http://vllm:8000/v1", "m").score(["Hi Ana"])[0]
    assert scored.offsets == [(0, 2), (2, 6)]  # the generated "!" after the prompt is dropped
    assert np.isnan(scored.logprobs[0])
    assert scored.logprobs[1] == -1.5


def test_anthropic_backend_needs_a_key_and_reads_text_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from model_passport.llm.backends import AnthropicModel, load_model

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ModelError, match="ANTHROPIC_API_KEY"):
        AnthropicModel().complete("hi")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _Response({"content": [{"type": "text", "text": "ok"}]})
    )
    model = load_model("anthropic:claude-sonnet-5")
    assert model.complete("hi", n=2) == ["ok", "ok"]  # type: ignore[union-attr]
    with pytest.raises(ModelError, match="unknown model spec"):
        load_model("nope:model")


# --- Passport integration and CLI ----------------------------------------------------------------


def test_policy_gate_blocks_critical_entities(sk_project: Path) -> None:
    import yaml

    from model_passport.core.builder import build_passport
    from model_passport.core.config import load_config

    result = EntityAuditResult(
        model="m", access="logprobs", primary_method="min_k", methods=["min_k"], references=5,
        entities_audited=100, controls=100, auc=0.9, tpr_at_fpr={"0.01": 0.3},
        severity_counts={"critical": 2, "high": 5},
    )  # fmt: skip
    (sk_project / "reports").mkdir()
    (sk_project / "reports/entity_audit.json").write_text(result.model_dump_json())
    policy = sk_project / "policy.yaml"
    policy.write_text(yaml.safe_dump({"rules": {
        "entity_critical_max": 0, "entity_high_max": {"warn": 0},
        "el_mia_auc_max": {"warn": 0.55, "fail": 0.60},
        "el_mia_tpr_at_1pct_fpr_max": {"warn": 0.02, "fail": 0.05}}}))  # fmt: skip
    config = yaml.safe_load((sk_project / "passport.yaml").read_text())
    config["policy"] = "policy.yaml"
    config.setdefault("privacy", {})["entity_audit"] = "reports/entity_audit.json"
    (sk_project / "passport.yaml").write_text(yaml.safe_dump(config))
    passport = build_passport(
        sk_project / "passport.yaml", load_config(sk_project / "passport.yaml")
    )
    rules = {r.name: r.result.value for r in passport.policy.rules}  # type: ignore[union-attr]
    assert rules == {
        "entity_critical_max": "fail",
        "entity_high_max": "warn",
        "el_mia_auc_max": "fail",
        "el_mia_tpr_at_1pct_fpr_max": "fail",
    }
    assert "reports/entity_audit.json" in {a.path for a in passport.artifacts}


def test_llm_cli_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    corpus = tmp_path / "corpus.jsonl"
    result = runner.invoke(app, ["llm", "demo-corpus", "--out", str(corpus), "--records", "80"])
    assert result.exit_code == 0, result.output
    model_dir = tmp_path / "model"
    result = runner.invoke(app, ["llm", "finetune", "--corpus", str(corpus), "--out",
                                 str(model_dir), "--epochs", "2", "--lr", "2e-3"])  # fmt: skip
    assert result.exit_code == 0, result.output
    report = tmp_path / "audit.json"
    result = runner.invoke(app, ["llm", "audit", "--model", str(model_dir), "--corpus",
                                 str(corpus), "--out", str(report), "--device", "cpu"])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "entities audited" in result.output
    audited = EntityAuditResult.model_validate_json(report.read_text())
    clean = tmp_path / "clean.jsonl"
    result = runner.invoke(app, ["llm", "sanitize", "--corpus", str(corpus), "--audit",
                                 str(report), "--always", "SSN", "--out", str(clean)])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert audited.entities_audited > 0
    assert "replaced" in result.output
    missing = runner.invoke(app, ["llm", "audit", "--model", "nope:x", "--corpus", str(corpus)])
    assert missing.exit_code != 0


def test_openai_chat_and_compatible_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from model_passport.llm.backends import OpenAICompatibleModel, load_model

    seen: list[str] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        seen.append(url)
        if url.endswith("/chat/completions"):
            return _Response({"choices": [{"message": {"content": "hi"}}] * kwargs["json"]["n"]})
        return _Response({"choices": [{"text": "there"}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert load_model("openai:gpt-5").complete("x", n=2) == ["hi", "hi"]  # type: ignore[union-attr]
    compatible = load_model("openai-compatible:http://vllm:8000/v1#llama")
    assert isinstance(compatible, OpenAICompatibleModel)
    assert compatible.complete("x") == ["there"]
    assert seen == ["https://api.openai.com/v1/chat/completions", "http://vllm:8000/v1/completions"]
    with pytest.raises(ModelError, match="openai-compatible"):
        load_model("openai-compatible:no-model-name")


def test_scoring_rejects_servers_without_prompt_logprobs(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from model_passport.llm.backends import OpenAICompatibleModel

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Response({"choices": [{"text": "x"}]}))
    with pytest.raises(ModelError, match="did not echo"):
        OpenAICompatibleModel("http://host/v1", "m").score(["text"])


def test_hugging_face_generation(trained: dict[str, Any]) -> None:
    outputs = trained["model"].complete("Please send the invoice to", n=2, max_tokens=8)
    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)


def test_unconfirmed_findings_are_capped_at_low(trained: dict[str, Any]) -> None:
    """With confirmation off, chance-level screens can report High; with it on they cannot."""
    from model_passport.llm import training

    records = trained["records"]
    unrelated = synthetic.corpus(300, seed=42)  # a model never trained on ``records``
    model = training.tiny_model([r.text for r in unrelated] + trained["filler"], seed=0)
    training.finetune(model, [r.text for r in unrelated], trained["settings"])
    confirmed = audit(model, records, AUDIT)
    assert confirmed.severity_counts["critical"] == 0
    assert confirmed.severity_counts["high"] == 0
    assert all(f.confirmed is False for f in confirmed.findings if f.confirmation_p is not None)
