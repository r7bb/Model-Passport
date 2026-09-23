from __future__ import annotations

import pandas as pd

from model_passport.scanners.base import ScanTarget
from model_passport.scanners.reid_risk import (
    ReidRiskScanner,
    assess,
    k_anonymity,
    l_diversity,
    risky_combinations,
    suggest_quasi_identifiers,
    unique_fraction,
)

FRAME = pd.DataFrame(
    {
        "age": [30, 30, 30, 40, 40, 50],
        "zip": ["021", "021", "021", "100", "100", "303"],
        "gender": ["F", "F", "M", "M", "M", "F"],
        "income": ["hi", "lo", "hi", "lo", "lo", "hi"],
    }
)


def test_k_anonymity_and_uniqueness() -> None:
    assert k_anonymity(FRAME, ["age", "zip"]) == 1  # the single 50/303 record
    assert k_anonymity(FRAME, ["zip"]) == 1
    assert k_anonymity(FRAME.iloc[:5], ["zip"]) == 2
    assert unique_fraction(FRAME, ["age", "zip", "gender"]) == 2 / 6


def test_l_diversity() -> None:
    assert l_diversity(FRAME.iloc[:5], ["zip"], "income") == 1  # zip 100 is all "lo"
    assert l_diversity(FRAME.iloc[:3], ["zip"], "income") == 2


def test_risky_combinations_are_minimal() -> None:
    combos = risky_combinations(FRAME, ["age", "zip", "gender"], threshold=0.0)
    column_sets = [set(c.columns) for c in combos]
    assert {"age"} in column_sets
    assert all(not ({"age"} < s) for s in column_sets)  # supersets of a risky set are skipped


def test_suggest_quasi_identifiers() -> None:
    frame = pd.DataFrame(columns=["Age", "zip_code", "Sex", "hours_per_week", "usage"])
    assert suggest_quasi_identifiers(frame) == ["Age", "zip_code", "Sex"]


def test_assess_and_scanner() -> None:
    result = assess(FRAME, "t", ["age", "zip", "gender"], "income")
    assert result.k_anonymity == 1 and result.l_diversity == 1 and result.rows == 6

    scanner = ReidRiskScanner(["age", "zip", "gender"])
    findings = scanner.scan(ScanTarget.from_frame(FRAME, "t"))
    assert findings[0].category == "K_ANONYMITY" and findings[0].severity.value == "critical"
    assert scanner.results[0].k_anonymity == 1


def test_no_quasi_identifiers() -> None:
    frame = pd.DataFrame({"x": [1, 2]})
    assert assess(frame, "t").k_anonymity is None
    assert ReidRiskScanner().scan(ScanTarget.from_frame(frame, "t")) == []
