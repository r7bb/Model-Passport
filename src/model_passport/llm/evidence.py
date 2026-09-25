"""Statistical evidence per entity: which attack to trust, and how surprising each score is.

**Strongest attack, without optimism.** An auditor should assume the strongest adversary, so
``primary="auto"`` uses whichever method separates members from controls best. Choosing it on
the same data it is then scored on would inflate the results, so the choice is cross-fitted:
entities are split into two folds, the method is chosen on one fold and applied to the other,
and the folds swap.

**Tail resolution.** An empirical p-value from n controls can never be below 1/(n + 1), too
coarse to survive a false-discovery correction over thousands of entities. As in LiRA (Carlini
et al., "Membership Inference Attacks From First Principles", 2022), the control scores are
modeled as a Gaussian, using the median and MAD so memorized outliers cannot distort it. Scores
are standardized against controls of the same entity type when there are enough, otherwise
against all controls in the fold.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from model_passport.llm.risk import attack_strength

MIN_TYPE_CONTROLS = 30  # to estimate a type's own center and spread
MAD_TO_SIGMA = 1.4826
FOLDS = 2


@dataclass
class Evidence:
    method_by_fold: list[str]
    z: np.ndarray  # member scores standardized against their controls
    control_z: np.ndarray
    p: np.ndarray

    @property
    def primary(self) -> str:
        chosen = sorted(set(self.method_by_fold))
        return chosen[0] if len(chosen) == 1 else "+".join(chosen)


def _robust(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    spread = MAD_TO_SIGMA * float(np.median(np.abs(values - center)))
    if not spread > 0:
        spread = float(np.std(values)) or 1.0
    return center, spread


def _standardize(
    indices: np.ndarray, types: Sequence[str], member: np.ndarray, control: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """z-scores of members and controls, by type where a type has enough controls."""
    z, cz = np.full(member.size, np.nan), np.full(control.size, np.nan)
    pooled = control[indices][np.isfinite(control[indices])]
    if not pooled.size:
        return z, cz
    fallback = _robust(pooled)
    by_type: dict[str, list[int]] = defaultdict(list)
    for i in indices:
        by_type[types[i]].append(int(i))
    for group in by_type.values():
        own = control[group][np.isfinite(control[group])]
        center, spread = _robust(own) if own.size >= MIN_TYPE_CONTROLS else fallback
        z[group] = (member[group] - center) / spread
        cz[group] = (control[group] - center) / spread
    return z, cz


def _best_method(
    methods: Sequence[str],
    member: Mapping[str, np.ndarray],
    control: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> str:
    def auc(name: str) -> float:
        value, _ = attack_strength(member[name][indices], control[name][indices])
        return value if np.isfinite(value) else 0.0

    return max(methods, key=auc)


def gather(
    types: Sequence[str],
    member: Mapping[str, Sequence[float]],
    control: Mapping[str, Sequence[float]],
    primary: str,
    seed: int = 0,
) -> Evidence:
    """Cross-fitted method choice (when ``primary == "auto"``) and Gaussian p-values."""
    methods = list(member)
    n = len(types)
    arrays = {k: np.asarray(v, dtype=float) for k, v in member.items()}
    controls = {k: np.asarray(v, dtype=float) for k, v in control.items()}
    order = list(range(n))
    random.Random(seed).shuffle(order)  # noqa: S311 - reproducible fold assignment
    folds = [np.asarray(sorted(order[f::FOLDS]), dtype=int) for f in range(FOLDS)]
    z, cz = np.full(n, np.nan), np.full(n, np.nan)
    chosen = []
    for f, fold in enumerate(folds):
        if not fold.size:
            continue
        other = np.concatenate([folds[g] for g in range(FOLDS) if g != f])
        if primary == "auto":
            method = _best_method(methods, arrays, controls, other if other.size else fold)
        else:
            method = primary
        chosen.append(method)
        fz, fcz = _standardize(fold, types, arrays[method], controls[method])
        z[fold], cz[fold] = fz[fold], fcz[fold]
    p = np.where(np.isfinite(z), norm.sf(np.nan_to_num(z, nan=0.0)), 1.0)
    return Evidence(chosen, z, cz, p)
