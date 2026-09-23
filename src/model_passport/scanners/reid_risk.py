"""Reidentification risk: k-anonymity, l-diversity, and uniqueness over quasi-identifiers."""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from model_passport.core.schema import (
    Finding,
    ReidentificationResult,
    RiskyCombination,
    Severity,
)
from model_passport.scanners.base import Scanner, ScanTarget
from model_passport.scanners.data_pii import column_tokens

QUASI_IDENTIFIER_HINTS = {
    "age", "zip", "zipcode", "zip_code", "postal", "postcode", "postal_code", "gender", "sex",
    "dob", "birth", "birthdate", "date_of_birth", "race", "ethnicity", "marital",
    "marital_status",
}  # fmt: skip


def suggest_quasi_identifiers(frame: pd.DataFrame) -> list[str]:
    """Columns whose names suggest they are quasi-identifiers."""
    return [str(c) for c in frame.columns if column_tokens(str(c)) & QUASI_IDENTIFIER_HINTS]


def _class_sizes(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame.groupby(columns, dropna=False, observed=True).size()


def unique_fraction(frame: pd.DataFrame, columns: list[str]) -> float:
    if frame.empty:
        return 0.0
    sizes = _class_sizes(frame, columns)
    return float((sizes == 1).sum()) / len(frame)


def k_anonymity(frame: pd.DataFrame, columns: list[str]) -> int | None:
    if frame.empty:
        return None
    return int(_class_sizes(frame, columns).min())


def l_diversity(frame: pd.DataFrame, columns: list[str], sensitive: str) -> int | None:
    """Distinct l-diversity: fewest distinct sensitive values in any equivalence class."""
    if frame.empty:
        return None
    grouped = frame.groupby(columns, dropna=False, observed=True)[sensitive]
    return int(grouped.nunique(dropna=False).min())


def risky_combinations(
    frame: pd.DataFrame, columns: list[str], max_size: int = 3, threshold: float = 0.01
) -> list[RiskyCombination]:
    """Minimal column subsets that make more than ``threshold`` of records unique."""
    risky: list[RiskyCombination] = []
    for size in range(1, min(max_size, len(columns)) + 1):
        for combo in combinations(columns, size):
            if any(set(r.columns) <= set(combo) for r in risky):
                continue  # a subset already makes records unique; report only minimal sets
            fraction = unique_fraction(frame, list(combo))
            if fraction > threshold:
                risky.append(RiskyCombination(columns=list(combo), unique_fraction=fraction))
    return sorted(risky, key=lambda r: -r.unique_fraction)


def assess(
    frame: pd.DataFrame,
    dataset: str,
    quasi_identifiers: list[str] | None = None,
    sensitive_column: str | None = None,
) -> ReidentificationResult:
    qis = quasi_identifiers if quasi_identifiers is not None else suggest_quasi_identifiers(frame)
    qis = [q for q in qis if q in frame.columns]
    if not qis:
        return ReidentificationResult(
            dataset=dataset, rows=len(frame), quasi_identifiers=[], k_anonymity=None,
            unique_fraction=0.0,
        )  # fmt: skip
    sensitive = sensitive_column if sensitive_column in frame.columns else None
    return ReidentificationResult(
        dataset=dataset,
        rows=len(frame),
        quasi_identifiers=qis,
        k_anonymity=k_anonymity(frame, qis),
        unique_fraction=round(unique_fraction(frame, qis), 6),
        sensitive_column=sensitive,
        l_diversity=l_diversity(frame, qis, sensitive) if sensitive else None,
        risky_combinations=[
            RiskyCombination(columns=r.columns, unique_fraction=round(r.unique_fraction, 6))
            for r in risky_combinations(frame, qis)
        ],
    )


def _severity(k: int | None) -> Severity:
    if k is None:
        return Severity.INFO
    if k <= 1:
        return Severity.CRITICAL
    if k < 5:
        return Severity.HIGH
    if k < 10:
        return Severity.MEDIUM
    return Severity.LOW


class ReidRiskScanner(Scanner):
    """Wraps ``assess`` in the common scanner interface."""

    name = "reid_risk"

    def __init__(
        self, quasi_identifiers: list[str] | None = None, sensitive_column: str | None = None
    ) -> None:
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_column = sensitive_column
        self.results: list[ReidentificationResult] = []

    def scan(self, target: ScanTarget) -> list[Finding]:
        result = assess(target.frame, target.label, self.quasi_identifiers, self.sensitive_column)
        self.results.append(result)
        if not result.quasi_identifiers:
            return []
        findings = [
            Finding(
                scanner=self.name,
                category="K_ANONYMITY",
                severity=_severity(result.k_anonymity),
                location=target.label,
                count=round(result.unique_fraction * result.rows),
                message=(
                    f"k={result.k_anonymity}, {result.unique_fraction:.1%} of records unique "
                    f"on {', '.join(result.quasi_identifiers)}"
                ),
                details={
                    "k_anonymity": result.k_anonymity,
                    "l_diversity": result.l_diversity,
                    "unique_fraction": result.unique_fraction,
                },
            )
        ]
        findings.extend(
            Finding(
                scanner=self.name,
                category="UNIQUE_COMBINATION",
                severity=Severity.MEDIUM if combo.unique_fraction < 0.05 else Severity.HIGH,
                location=f"{target.label}:{'+'.join(combo.columns)}",
                count=round(combo.unique_fraction * result.rows),
                message=(
                    f"{', '.join(combo.columns)} makes {combo.unique_fraction:.1%} "
                    "of records unique"
                ),
                details={"unique_fraction": combo.unique_fraction},
            )
            for combo in result.risky_combinations
        )
        return findings
