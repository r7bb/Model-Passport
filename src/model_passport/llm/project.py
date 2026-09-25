"""Language model projects: the files ``passport init --llm`` writes.

Layout::

    data/raw.jsonl        the original corpus; audits test its values, proving they are gone
    data/train.jsonl      what the model trains on; remediation sanitizes it in place
    data/history/         training data of earlier versions
    models/model/         the fine-tuned checkpoint (safetensors)
    reports/entity_audit.json
    passport.yaml, policy.yaml, stages/finetune.py, stages/audit.py
"""

from __future__ import annotations

from pathlib import Path

from model_passport.llm.entities import load_corpus, write_corpus

RAW_CORPUS = "data/raw.jsonl"
TRAIN_CORPUS = "data/train.jsonl"
HISTORY_DIR = "data/history"
MODEL_DIR = "models/model"
AUDIT_REPORT = "reports/entity_audit.json"

CONFIG_TEMPLATE = """\
# MP project for a language model trained on {source}.
# Created by `passport init --llm`. Next: `passport run && passport build`, then
# `passport llm remediate` if the release gate fails.

project:
  name: {name}
  version: 1.0.0

signing:
  private_key: .passport/signing_key.pem   # never commit this file
  public_key: .passport/signing_key.pub

stages:
  - name: finetune
    cmd: python stages/finetune.py
    params:
      base: {base}                 # hf:<model id>, a local checkpoint, or tiny (demo)
      corpus: {train}
      out: {model}
      epochs: {epochs}
      learning_rate: {learning_rate}
      batch_size: {batch_size}
      seed: 0
    deps: [{train}]
    outs: [{model}]

  - name: audit
    cmd: python stages/audit.py
    params:
      model: {model}
      corpus: {raw}                # the original data: its values must not be memorized
      out: {report}
      references: 5                # look-alike values per entity (EL-MIA uses 5)
      source: synthetic            # synthetic (strongest test) | corpus | mix
      max_entities: 2000
      fdr: 0.05
    deps: [{model}, {raw}]
    outs: [{report}]

declared:                          # shown in the report; fill these in
  intended_use: null
  out_of_scope_uses: []
  known_limitations: []
  ethical_risks: []
  owner: null
  contact: null

build:
  model:
    path: {model}
    framework: transformers
    task_type: text-generation
  datasets:
    - path: {train}
      split_role: train
      source: {source}

privacy:
  enabled: false                   # table scans do not apply to text; the entity audit does
  entity_audit: {report}

policy: policy.yaml
"""

POLICY_TEMPLATE = """\
# MP release gate for language models. `passport build` exits 1 when the verdict is fail.
#
# Defaults follow industry practice:
# - Confirmed Critical and High findings block release, as in vulnerability management
#   (CVSS severity bands).
# - Attack strength is judged by AUC (0.5 = chance) and by the true-positive rate at a 1%
#   false-positive rate (0.01 = chance), as in Carlini et al. 2022.

rules:
  entity_critical_max: 0                                 # confirmed Critical findings
  entity_high_max: 0                                     # confirmed High findings
  el_mia_auc_max: {warn: 0.55, fail: 0.60}
  el_mia_tpr_at_1pct_fpr_max: {warn: 0.02, fail: 0.05}
  secrets_found_max: 0
  unsafe_pickle: fail
  critical_cves_max: 0

missing_evidence: warn
"""

STAGE_SCRIPT = '''\
"""{stage} stage created by `passport init --llm`. See model_passport.llm.stages."""

from model_passport.llm.stages import run

if __name__ == "__main__":
    run("{stage}")
'''


# Fine-tuning a real checkpoint: a few gentle epochs. A tiny demo model starts from scratch,
# so it needs many more steps at a higher rate to learn (and memorize) its corpus.
TRAINING = {"checkpoint": (3, 5e-5, 8), "tiny": (15, 2e-3, 32)}


def scaffold(
    root: Path, name: str, corpus: Path, base: str = "tiny", force: bool = False
) -> list[str]:
    """Write a language model project; returns the files written (existing ones are kept)."""
    epochs, learning_rate, batch_size = TRAINING["tiny" if base == "tiny" else "checkpoint"]
    records = list(load_corpus(corpus))
    if not any(r.entities for r in records):
        raise ValueError(f"no sensitive entities in {corpus}; annotate them or check the format")
    written: list[str] = []

    def put(relative: str, content: str) -> None:
        path = root / relative
        if path.exists() and not force:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(relative)

    for target in (RAW_CORPUS, TRAIN_CORPUS):
        if force or not (root / target).exists():
            write_corpus(records, root / target)
            written.append(target)
    put("passport.yaml", CONFIG_TEMPLATE.format(
        name=name, source=corpus.name, base=base, train=TRAIN_CORPUS, raw=RAW_CORPUS,
        model=MODEL_DIR, report=AUDIT_REPORT, epochs=epochs, learning_rate=learning_rate,
        batch_size=batch_size,
    ))  # fmt: skip
    put("policy.yaml", POLICY_TEMPLATE)
    for stage in ("finetune", "audit"):
        put(f"stages/{stage}.py", STAGE_SCRIPT.format(stage=stage))
    return written
