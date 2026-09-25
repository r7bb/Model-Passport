"""Turn membership scores into per-entity risk, following common industry practice.

- **Evidence per entity:** a p-value against controls, meaning same-type values the model
  never saw in that slot (see ``model_passport.llm.evidence``). Auditing thousands of entities
  makes some look extreme by chance, so p-values are adjusted with the Benjamini-Hochberg false
  discovery rate. Only entities that stay significant can rate above Low.
- **Risk:** likelihood x impact, as in NIST SP 800-30, scaled to 0-10. Likelihood is 1 - p;
  impact comes from the entity type (see ``model_passport.llm.entities``).
- **Severity:** the CVSS v3.1 / v4.0 qualitative bands: Low 0.1-3.9, Medium 4.0-6.9,
  High 7.0-8.9, Critical 9.0-10.0.
- **Attack strength:** AUC plus the true-positive rate at low false-positive rates (0.1%, 1%,
  5%), which is how membership inference is judged (Carlini et al., "Membership Inference
  Attacks From First Principles", IEEE S&P 2022).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from model_passport.core.schema import Severity

FPR_LEVELS = (0.001, 0.01, 0.05)
NOT_SIGNIFICANT_CAP = 3.9  # at most Low without statistical evidence
BANDS = (
    (9.0, Severity.CRITICAL),
    (7.0, Severity.HIGH),
    (4.0, Severity.MEDIUM),
    (0.1, Severity.LOW),
)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Adjusted p-values (q-values) controlling the false discovery rate."""
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def severity(risk: float) -> Severity:
    for floor, level in BANDS:
        if risk >= floor:
            return level
    return Severity.INFO


def entity_risk(p: float, q: float, impact: float, fdr: float) -> tuple[float, float]:
    """(likelihood, risk on 0-10)."""
    likelihood = 1.0 - p
    risk = 10.0 * likelihood * impact
    if q > fdr:
        risk = min(risk, NOT_SIGNIFICANT_CAP)
    return likelihood, round(risk, 2)


def attack_strength(
    members: Sequence[float] | np.ndarray,
    controls: Sequence[float] | np.ndarray,
    levels: Sequence[float] = FPR_LEVELS,
) -> tuple[float, dict[str, float]]:
    """(AUC, {fpr: tpr}) of separating members from controls; NaN scores are dropped."""
    member = np.asarray(members, dtype=float)
    control = np.asarray(controls, dtype=float)
    member, control = member[~np.isnan(member)], control[~np.isnan(control)]
    if not member.size or not control.size:
        return float("nan"), {}
    labels = np.concatenate([np.ones(member.size), np.zeros(control.size)])
    scores = np.concatenate([member, control])
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    rates = {}
    for level in levels:
        allowed = tpr[fpr <= level]
        rates[f"{level:g}"] = round(float(allowed.max()) if allowed.size else 0.0, 4)
    return round(auc, 4), rates
