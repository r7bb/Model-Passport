"""Measure the entity audit against ground truth: a real open model on public labeled data.

Documents from Nemotron-PII (CC BY 4.0) are split by id into three groups: **trained** (the
model is fine-tuned on them, each repeated 1, 2, 4, or 8 times, since memorization grows
with repetition), **calibration** (held out; the audit's real non-members), and **unseen**
(never trained on, and audited as if they were training data). Every flagged unseen entity
is therefore a false alarm, and every trained entity should ideally be caught.

The same audits run on the model before fine-tuning, where nothing can have been memorized
from these documents, so any finding is a false alarm. Each audit runs twice: calibrated
with synthetic look-alikes, and with the held-out real documents.

    python scripts/benchmark.py --model EleutherAI/pythia-160m --docs 1200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_passport.llm import audit as audit_module
from model_passport.llm import training
from model_passport.llm.audit import AuditSettings, audit, targets
from model_passport.llm.backends import HuggingFaceModel, Scored
from model_passport.llm.entities import Record
from model_passport.llm.public_data import (
    NEMOTRON_ATTRIBUTION,
    download_nemotron,
    nemotron_records,
)

REPO = Path(__file__).resolve().parents[1]
REPEATS = (1, 2, 4, 8)


def _bucket(record_id: str, salt: str, size: int) -> int:
    digest = hashlib.sha256(f"{salt}:{record_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % size


@dataclass
class Groups:
    trained: list[Record]
    calibration: list[Record]
    unseen: list[Record]
    repeats: dict[str, int]


def split(records: Sequence[Record]) -> Groups:
    """60% trained, 20% calibration, 20% unseen; trained records get a repeat count."""
    groups = Groups([], [], [], {})
    for record in records:
        slot = _bucket(record.id, "group", 5)
        if slot < 3:
            groups.trained.append(record)
            groups.repeats[record.id] = REPEATS[_bucket(record.id, "repeat", len(REPEATS))]
        elif slot == 3:
            groups.calibration.append(record)
        else:
            groups.unseen.append(record)
    return groups


class CachedModel:
    """Remembers scores by text, so repeated audits of the same model are almost free."""

    def __init__(self, model: HuggingFaceModel) -> None:
        self.model = model
        self.name = model.name
        self.cache: dict[str, Scored] = {}

    def score(self, texts: Sequence[str]) -> list[Scored]:
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            self.cache.update(zip(missing, self.model.score(missing), strict=True))
        return [self.cache[t] for t in texts]


def _rates(hits: int, total: int) -> dict[str, Any]:
    return {"flagged": hits, "of": total, "rate": round(hits / total, 4) if total else None}


def measure(
    model: CachedModel, groups: Groups, settings: AuditSettings, calibrated: bool
) -> dict[str, Any]:
    """Detection among trained entities and false alarms among unseen ones."""
    records = groups.trained + groups.unseen
    start = time.monotonic()
    result = audit(model, records, settings, holdout=groups.calibration if calibrated else ())
    audited = targets(records, settings)  # the same sample the audit used
    trained_ids = set(groups.repeats)
    totals: dict[str, int] = defaultdict(int)
    for target in audited:
        rid = target.record.id
        totals[f"k{groups.repeats[rid]}" if rid in trained_ids else "unseen"] += 1
    significant: dict[str, int] = defaultdict(int)
    high: dict[str, int] = defaultdict(int)
    for finding in result.findings:
        rid = finding.record
        key = f"k{groups.repeats[rid]}" if rid in trained_ids else "unseen"
        if finding.q_value <= settings.fdr:
            significant[key] += 1
        if finding.severity.value in ("high", "critical"):
            high[key] += 1
    trained_total = sum(totals[f"k{k}"] for k in REPEATS)
    return {
        "calibration": result.calibration,
        "seconds": round(time.monotonic() - start, 1),
        "primary_method": result.primary_method,
        "holdout_auc": result.holdout_auc,
        "control_shift_auc": result.control_shift_auc,
        "detected": _rates(sum(significant[f"k{k}"] for k in REPEATS), trained_total),
        "detected_by_repeats": {
            str(k): _rates(significant[f"k{k}"], totals[f"k{k}"]) for k in REPEATS
        },
        "false_alarms": _rates(significant["unseen"], totals["unseen"]),
        "high_or_critical": {
            "trained": _rates(sum(high[f"k{k}"] for k in REPEATS), trained_total),
            "unseen": _rates(high["unseen"], totals["unseen"]),
        },
    }


def _log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="EleutherAI/pythia-160m")
    parser.add_argument("--docs", type=int, default=1200)
    parser.add_argument("--max-chars", type=int, default=600)
    parser.add_argument("--entities", type=int, default=1200, help="entities per audit")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--data", type=Path, default=REPO / ".cache" / "public-data")
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "benchmarks")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    audit_module.MAX_FINDINGS = sys.maxsize  # the benchmark scores every finding
    path = download_nemotron(args.data)
    records = list(nemotron_records(path, limit=args.docs, max_chars=args.max_chars))
    groups = split(records)
    _log(
        f"{len(records)} documents: {len(groups.trained)} trained, "
        f"{len(groups.calibration)} calibration, {len(groups.unseen)} unseen"
    )
    settings = AuditSettings(max_entities=args.entities, seed=0)
    base = HuggingFaceModel.load(args.model, args.device)
    report: dict[str, Any] = {
        "model": args.model,
        "data": NEMOTRON_ATTRIBUTION,
        "documents": {
            "trained": len(groups.trained),
            "calibration": len(groups.calibration),
            "unseen": len(groups.unseen),
        },
        "fine_tuning": {"epochs": args.epochs, "learning_rate": args.lr, "repeats": REPEATS},
        "fdr": settings.fdr,
    }

    model = CachedModel(base)
    for name, calibrated in (("look_alikes", False), ("held_out", True)):
        _log(f"before fine-tuning, {name} …")
        report.setdefault("before_fine_tuning", {})[name] = measure(
            model, groups, settings, calibrated
        )

    texts = [r.text for r in groups.trained for _ in range(groups.repeats[r.id])]
    _log(f"fine-tuning on {len(texts)} sequences …")
    start = time.monotonic()
    tuning = training.TrainSettings(
        epochs=args.epochs, learning_rate=args.lr, batch_size=8, device=args.device
    )
    losses = training.finetune(base, texts, tuning)
    report["fine_tuning"] |= {
        "loss": [round(x, 4) for x in losses],
        "seconds": round(time.monotonic() - start, 1),
    }
    base.model.eval()

    model = CachedModel(base)  # new weights: forget the old scores
    for name, calibrated in (("look_alikes", False), ("held_out", True)):
        _log(f"after fine-tuning, {name} …")
        report.setdefault("after_fine_tuning", {})[name] = measure(
            model, groups, settings, calibrated
        )

    args.out.mkdir(parents=True, exist_ok=True)
    slug = args.model.split("/")[-1].lower()
    target = args.out / f"nemotron-{slug}.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _log(json.dumps(report, indent=2))
    _log(f"wrote {target}")


if __name__ == "__main__":
    main()
