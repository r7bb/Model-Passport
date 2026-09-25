"""Pipeline stages for language model projects: fine-tune, then audit.

``passport init --llm`` writes two small scripts that call ``run``; ``passport run`` executes
them with the parameters from ``passport.yaml``. The same functions back the
``passport llm finetune`` and ``passport llm audit`` commands.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from model_passport.core.schema import EntityAuditResult
from model_passport.llm import training
from model_passport.llm.audit import AuditSettings, audit
from model_passport.llm.backends import HuggingFaceModel, ModelError, load_model
from model_passport.llm.entities import Record, load_corpus
from model_passport.runtime import params

FINGERPRINT_ENV = "PASSPORT_FINGERPRINT_KEY"

FINETUNE_DEFAULTS: dict[str, Any] = {
    "base": "tiny",  # a Hugging Face id (hf:...) or directory; "tiny" builds a demo model
    "corpus": "data/train.jsonl",
    "out": "models/model",
    "epochs": 3,
    "learning_rate": 5e-5,
    "batch_size": 8,
    "seed": 0,
    "device": None,
}
AUDIT_DEFAULTS: dict[str, Any] = {
    "model": "models/model",
    "corpus": "data/raw.jsonl",  # the original data: proves its values are not memorized
    "out": "reports/entity_audit.json",
    "references": 5,
    "source": "synthetic",
    "max_entities": 2000,
    "primary": "auto",
    "fdr": 0.05,
    "probe_samples": 5,
    "types": [],
    "device": None,
}


class StageError(RuntimeError):
    """A stage cannot run; the message says what to fix."""


def read_corpus(path: Path) -> list[Record]:
    if not path.is_file():
        raise StageError(f"corpus not found: {path}")
    records = list(load_corpus(path))
    if not any(r.entities for r in records):
        raise StageError(f"no sensitive entities in {path}; annotate them or check the format")
    return records


def finetune_corpus(
    corpus: Path, out: Path, base: str, settings: training.TrainSettings
) -> list[float]:
    """Fine-tune ``base`` (or a new tiny model) on ``corpus`` and save it to ``out``."""
    texts = [r.text for r in read_corpus(corpus)]
    if base == "tiny":
        model = training.tiny_model(texts, seed=settings.seed)
    else:
        loaded = load_model(base, settings.device)
        if not isinstance(loaded, HuggingFaceModel):
            raise StageError("fine-tuning needs a Hugging Face checkpoint (hf:<id> or a directory)")
        model = loaded
    history = training.finetune(model, texts, settings)
    training.save(model, out)
    return history


def audit_corpus(
    model_spec: str, corpus: Path, out: Path, settings: AuditSettings, device: str | None = None
) -> EntityAuditResult:
    """Audit a model on the entities of ``corpus`` and write the result to ``out``."""
    records = read_corpus(corpus)
    result = audit(load_model(model_spec, device), records, settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result


def fingerprint_key() -> bytes | None:
    key = os.environ.get(FINGERPRINT_ENV)
    return key.encode() if key else None


def _finetune(p: Mapping[str, Any]) -> str:
    settings = training.TrainSettings(
        epochs=p["epochs"], learning_rate=p["learning_rate"], batch_size=p["batch_size"],
        seed=p["seed"], device=p["device"],
    )  # fmt: skip
    history = finetune_corpus(Path(p["corpus"]), Path(p["out"]), p["base"], settings)
    losses = f"{history[0]:.3f} -> {history[-1]:.3f}"
    return f"fine-tuned {p['base']} for {p['epochs']} epochs; loss {losses}"


def _audit(p: Mapping[str, Any]) -> str:
    settings = AuditSettings(
        references=p["references"], reference_source=p["source"], max_entities=p["max_entities"],
        primary=p["primary"], fdr=p["fdr"], probe_samples=p["probe_samples"],
        types=tuple(p["types"]), fingerprint_key=fingerprint_key(),
    )  # fmt: skip
    result = audit_corpus(p["model"], Path(p["corpus"]), Path(p["out"]), settings, p["device"])
    counts = result.severity_counts
    auc = "n/a" if result.auc is None else f"{result.auc:.3f}"
    return (
        f"audited {result.entities_audited} entities: AUC {auc}; critical "
        f"{counts.get('critical', 0)}, high {counts.get('high', 0)}, "
        f"medium {counts.get('medium', 0)}"
    )


STAGES = {"finetune": (FINETUNE_DEFAULTS, _finetune), "audit": (AUDIT_DEFAULTS, _audit)}


def run(stage: str) -> None:
    """Run one stage with parameters from ``passport run``; exit with a message on bad input."""
    defaults, function = STAGES[stage]
    try:
        summary = function(params(defaults))
    except (StageError, ModelError, ValueError) as exc:
        raise SystemExit(f"{stage}: {exc}") from exc
    sys.stdout.write(summary + "\n")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "")
