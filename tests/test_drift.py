from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_passport.monitoring.drift import (
    check_drift,
    check_schema,
    psi_categorical,
    psi_numeric,
)


def _frame(n: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    color_p = [0.5 - 0.3 * shift, 0.3, 0.2 + 0.3 * shift]
    return pd.DataFrame(
        {
            "hours": rng.normal(40 + 10 * shift, 10, n),
            "income": rng.lognormal(10, 0.5, n),
            "color": rng.choice(["red", "green", "blue"], n, p=color_p),
            "label": rng.integers(0, 2, n),
        }
    )


def test_same_distribution_has_no_drift() -> None:
    report = check_drift(_frame(3000, 1), _frame(1000, 2), exclude=["label"])
    assert not report.drift_detected
    assert report.schema.ok
    assert {f.feature for f in report.features} == {"hours", "income", "color"}
    assert all(f.psi < 0.1 for f in report.features)


def test_shifted_features_are_detected_and_others_are_not() -> None:
    reference = _frame(3000, 1)
    batch = _frame(1000, 2, shift=1.0).assign(income=_frame(1000, 3)["income"])
    report = check_drift(reference, batch, exclude=["label"])
    assert set(report.drifted_features) == {"hours", "color"}
    hours = next(f for f in report.features if f.feature == "hours")
    assert hours.kind == "numeric"
    assert hours.rejected == 3
    assert set(hours.tests) == {"kolmogorov_smirnov", "mann_whitney_u", "cramer_von_mises"}
    color = next(f for f in report.features if f.feature == "color")
    assert color.kind == "categorical"
    assert set(color.tests) == {"chi_square"}


def test_tiny_but_significant_shift_is_not_drift() -> None:
    """With huge samples a trivial shift is significant; the PSI gate must suppress it."""
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 200_000)})
    batch = pd.DataFrame({"x": rng.normal(0.03, 1, 200_000)})
    (feature,) = check_drift(reference, batch).features
    assert feature.rejected >= 2
    assert not feature.drifted


def test_label_and_features_selection() -> None:
    report = check_drift(_frame(500, 1), _frame(500, 2), features=["hours", "label"],
                         exclude=["label"])  # fmt: skip
    assert [f.feature for f in report.features] == ["hours"]


def test_schema_problems() -> None:
    reference = _frame(500, 1)
    batch = (
        _frame(500, 2)
        .drop(columns=["income"])
        .assign(hours=lambda d: d["hours"].astype(str), extra=1)
    )
    batch.loc[:200, "color"] = None
    check = check_schema(reference, batch, ["hours", "income", "color"])
    assert check.missing_columns == ["income"]
    assert check.unexpected_columns == ["extra"]
    assert check.type_mismatches == ["hours"]
    assert "color" in check.null_rate_increases
    assert not check.ok

    report = check_drift(reference, batch, exclude=["label"])
    assert "hours" not in {f.feature for f in report.features}  # untestable, reported in schema
    payload = report.to_payload()
    assert payload["schema_ok"] is False
    assert payload["schema"]["missing_columns"] == ["income"]


def test_unseen_categories_counted() -> None:
    reference = pd.DataFrame({"c": ["a", "b"] * 200})
    batch = pd.DataFrame({"c": ["a", "b", "z"] * 100})
    (feature,) = check_drift(reference, batch).features
    assert feature.unseen_categories == 1
    assert feature.drifted


def test_psi_properties() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=5000)
    assert psi_numeric(x, x) == pytest.approx(0.0, abs=1e-9)
    assert psi_numeric(x, x + 1.0) > 0.25
    assert psi_numeric(np.zeros(100), np.ones(100)) == 0.0  # constant reference: no bins
    s = pd.Series(["a", "b"] * 50)
    assert psi_categorical(s, s) == pytest.approx(0.0, abs=1e-9)
