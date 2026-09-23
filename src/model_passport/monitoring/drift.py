"""Data drift and schema checks between a training reference and a live batch.

Numeric features use Kolmogorov-Smirnov, Mann-Whitney U, and Cramér-von Mises tests;
categorical features use a chi-square test of homogeneity. Significance alone flags trivial
shifts on large batches, so a feature is *drifted* only when the tests reject (Bonferroni
corrected across features) **and** the population stability index shows a material effect.

Column types come from the same inference the trainer uses: numbers stored as text are compared
as numbers, dates as days, and identifier or free-text columns are skipped. Categories outside
the reference's most common ones are pooled, so a long tail cannot inflate the PSI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from model_passport.ml.features import Role, column_role, datetime_days, numeric_values

DEFAULT_ALPHA = 0.01
DEFAULT_PSI_THRESHOLD = 0.1  # 0.1-0.25 moderate shift, > 0.25 major shift
PSI_BINS = 10
NULL_RATE_TOLERANCE = 0.1
EPS = 1e-6
MAX_CATEGORIES = 20  # reference categories kept as-is; the rest are pooled as OTHER
OTHER = "__other__"


@dataclass
class FeatureDrift:
    feature: str
    kind: str
    psi: float
    tests: dict[str, dict[str, float]]
    rejected: int
    drifted: bool
    unseen_categories: int = 0


@dataclass
class SchemaCheck:
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    type_mismatches: list[str] = field(default_factory=list)
    null_rate_increases: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (self.missing_columns or self.type_mismatches or self.null_rate_increases)


@dataclass
class DriftReport:
    rows: int
    reference_rows: int
    alpha: float
    psi_threshold: float
    schema: SchemaCheck
    features: list[FeatureDrift]

    @property
    def drifted_features(self) -> list[str]:
        return [f.feature for f in self.features if f.drifted]

    @property
    def drift_detected(self) -> bool:
        return bool(self.drifted_features)

    def to_payload(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "reference_rows": self.reference_rows,
            "alpha": self.alpha,
            "psi_threshold": self.psi_threshold,
            "drift_detected": self.drift_detected,
            "drifted_features": self.drifted_features,
            "schema_ok": self.schema.ok,
            "schema": asdict(self.schema),
            "features": {f.feature: asdict(f) for f in self.features},
        }


def psi_numeric(reference: np.ndarray, batch: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population stability index over reference quantile bins."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(batch, edges)[0] / len(batch)
    return _psi(ref, cur)


def psi_categorical(reference: pd.Series, batch: pd.Series) -> float:
    categories = sorted(set(reference) | set(batch), key=str)
    ref = reference.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    cur = batch.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    return _psi(ref, cur)


def _psi(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _numeric_tests(reference: np.ndarray, batch: np.ndarray) -> dict[str, dict[str, float]]:
    ks = stats.ks_2samp(reference, batch)
    mwu = stats.mannwhitneyu(reference, batch, alternative="two-sided")
    cvm = stats.cramervonmises_2samp(reference, batch)
    return {
        "kolmogorov_smirnov": {"statistic": float(ks.statistic), "p_value": float(ks.pvalue)},
        "mann_whitney_u": {"statistic": float(mwu.statistic), "p_value": float(mwu.pvalue)},
        "cramer_von_mises": {"statistic": float(cvm.statistic), "p_value": float(cvm.pvalue)},
    }


def _chi_square(reference: pd.Series, batch: pd.Series) -> dict[str, dict[str, float]]:
    categories = sorted(set(reference) | set(batch), key=str)
    table = np.array(
        [
            reference.value_counts().reindex(categories, fill_value=0).to_numpy(),
            batch.value_counts().reindex(categories, fill_value=0).to_numpy(),
        ]
    )
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return {"chi_square": {"statistic": 0.0, "p_value": 1.0}}
    result = stats.chi2_contingency(table)
    return {"chi_square": {"statistic": float(result.statistic), "p_value": float(result.pvalue)}}


def check_schema(reference: pd.DataFrame, batch: pd.DataFrame, columns: list[str]) -> SchemaCheck:
    check = SchemaCheck(
        missing_columns=[c for c in columns if c not in batch.columns],
        unexpected_columns=[c for c in batch.columns if c not in reference.columns],
    )
    for column in columns:
        if column not in batch.columns:
            continue
        ref_numeric = column_role(column, reference[column]) is Role.NUMERIC
        batch_values = batch[column].dropna()
        parse_rate = numeric_values(batch_values).notna().mean() if len(batch_values) else 1.0
        if ref_numeric and parse_rate < 0.95:
            check.type_mismatches.append(column)
        increase = float(batch[column].isna().mean() - reference[column].isna().mean())
        if increase > NULL_RATE_TOLERANCE:
            check.null_rate_increases[column] = round(increase, 4)
    return check


def _pool(reference: pd.Series, batch: pd.Series) -> tuple[pd.Series, pd.Series]:
    keep = set(reference.value_counts().index[:MAX_CATEGORIES])
    return (
        reference.where(reference.isin(keep), OTHER),
        batch.where(batch.isin(keep), OTHER),
    )


def _compare(column: str, reference: pd.Series, batch: pd.Series) -> FeatureDrift | None:
    """Tests and PSI for one column, or None when the column cannot be compared."""
    role = column_role(column, reference)
    if not isinstance(role, Role) and role not in {"constant"}:
        return None  # identifiers, free text, and empty columns carry no distribution
    if role is Role.DATETIME:
        ref, cur = datetime_days(reference).dropna(), datetime_days(batch).dropna()
    elif role is Role.NUMERIC or (role == "constant" and pd.api.types.is_numeric_dtype(reference)):
        ref, cur = numeric_values(reference).dropna(), numeric_values(batch).dropna()
    else:
        ref, cur = _pool(reference.dropna().astype(str), batch.dropna().astype(str))
        unseen = len(set(batch.dropna().astype(str)) - set(reference.dropna().astype(str)))
        if ref.empty or cur.empty:
            return None
        return _feature(column, "categorical", _chi_square(ref, cur), psi_categorical(ref, cur),
                        unseen)  # fmt: skip
    if ref.empty or cur.empty:
        return None
    ref_values, cur_values = ref.to_numpy(float), cur.to_numpy(float)
    tests = _numeric_tests(ref_values, cur_values)
    kind = "datetime" if role is Role.DATETIME else "numeric"
    return _feature(column, kind, tests, psi_numeric(ref_values, cur_values), 0)


def _feature(
    column: str, kind: str, tests: dict[str, dict[str, float]], psi: float, unseen: int
) -> FeatureDrift:
    # ``drifted`` is decided by check_drift once the corrected alpha is known.
    return FeatureDrift(column, kind, round(psi, 6), tests, 0, False, unseen)


def check_drift(
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    features: list[str] | None = None,
    exclude: list[str] | None = None,
    alpha: float = DEFAULT_ALPHA,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
) -> DriftReport:
    excluded = set(exclude or [])
    columns = [c for c in (features or list(reference.columns)) if c not in excluded]
    schema = check_schema(reference, batch, columns)
    testable = [c for c in columns if c in batch.columns and c not in schema.type_mismatches]
    compared = [_compare(str(c), reference[c], batch[c]) for c in testable]
    results = [r for r in compared if r is not None]
    corrected_alpha = alpha / max(len(results), 1)
    for result in results:
        result.rejected = sum(t["p_value"] < corrected_alpha for t in result.tests.values())
        majority = result.rejected * 2 > len(result.tests)
        result.drifted = majority and result.psi >= psi_threshold
    return DriftReport(
        rows=len(batch),
        reference_rows=len(reference),
        alpha=alpha,
        psi_threshold=psi_threshold,
        schema=schema,
        features=results,
    )
