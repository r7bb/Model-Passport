"""Held-out calibration: findings must reflect memorization, not how real values look.

The fake model below has memorized nobody. It only finds a company's email domain familiar, as
a model trained on public text would. Synthetic controls use other domains, so against them
every real email looks "memorized"; against real emails the model never trained on, none do.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from model_passport.llm.audit import AuditSettings, audit
from model_passport.llm.backends import Scored
from model_passport.llm.entities import Record, detect, split_holdout

pytest.importorskip("faker")
DOMAIN = "corpmail.test"


class DomainFamiliarModel:
    """One token per character; characters of the company domain are highly likely."""

    name = "domain-familiar"

    def score(self, texts: Sequence[str]) -> list[Scored]:
        out = []
        for text in texts:
            # Ordinary characters vary a little in likelihood, as real tokens do.
            logprobs = np.array([-3.0 + 0.1 * (ord(c) % 7) for c in text])
            start = text.find(DOMAIN)
            while start != -1:
                logprobs[start : start + len(DOMAIN)] = -0.1
                start = text.find(DOMAIN, start + 1)
            logprobs[0] = np.nan
            out.append(Scored([(i, i + 1) for i in range(len(text))], logprobs))
        return out


def _records(start: int, count: int) -> list[Record]:
    records = []
    for i in range(start, start + count):
        text = f"Please forward the invoice to staff{i}@{DOMAIN} before Friday."
        records.append(Record(f"r{i}", text, detect(text)))
    return records


def _significant(result_findings: Sequence[object], fdr: float) -> int:
    return sum(1 for f in result_findings if getattr(f, "q_value", 1.0) <= fdr)


def test_synthetic_bias_is_caught_by_held_out_entities() -> None:
    members, held = split_holdout(_records(100, 200), 0.4)
    assert all(r.entities for r in members)
    assert len(held) >= 60
    settings = AuditSettings(confirm=False, seed=1)
    model = DomainFamiliarModel()

    naive = audit(model, members, settings)
    assert naive.calibration == "synthetic"
    assert (
        _significant(naive.findings, settings.fdr) > len(members) // 2
    )  # every email looks "memorized"

    calibrated = audit(model, members, settings, holdout=held)
    assert calibrated.calibration == "holdout"
    assert calibrated.holdout_entities == len(held)
    assert calibrated.control_shift_auc is not None
    assert calibrated.control_shift_auc > 0.9
    assert calibrated.holdout_auc is not None
    assert abs(calibrated.holdout_auc - 0.5) < 0.15
    assert _significant(calibrated.findings, settings.fdr) <= 3


def test_too_few_held_out_entities_fall_back_to_synthetic_controls() -> None:
    result = audit(
        DomainFamiliarModel(),
        _records(0, 40),
        AuditSettings(confirm=False),
        holdout=_records(500, 5),
    )
    assert result.calibration == "synthetic"
    assert result.holdout_auc is None


def test_holdout_split_is_stable_and_follows_ids() -> None:
    records = _records(0, 1000)
    train, held = split_holdout(records, 0.15)
    assert len(train) + len(held) == 1000
    assert 100 < len(held) < 200
    renamed = [Record(r.id, r.text.upper(), r.entities) for r in records]  # e.g. sanitized
    assert [r.id for r in split_holdout(renamed, 0.15)[1]] == [r.id for r in held]
    assert split_holdout(records, 0.0) == (records, [])
