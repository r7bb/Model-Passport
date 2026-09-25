"""Adaptive training: any table, any task, messy inputs, and honest model selection."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from faker import Faker
from sklearn import datasets
from sklearn.linear_model import LogisticRegression, Ridge

from model_passport.auditors.artifact import safe_load_pickle
from model_passport.auditors.leakage import loss_threshold_attack
from model_passport.ml import (
    Budget,
    DataError,
    Family,
    Role,
    SearchSettings,
    Task,
    clean_rows,
    evaluate,
    fit_fixed,
    infer_plan,
    infer_task,
    read_table,
    select_model,
    split,
)
from model_passport.ml.models import Trial, choose
from model_passport.ml.privacy import (
    apply_generalization,
    detect_identifiers,
    fit_generalization,
    range_label,
)

QUICK = SearchSettings(budget=Budget.QUICK, seed=0)


def messy_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Every awkward column type at once, with a learnable label."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    city = rng.choice(["Boston", "Austin", "Denver", None], n)
    signal = x + (city == "Austin") + rng.normal(0, 0.5, n)
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "x_text": [f"{v:.3f}" for v in x],  # numbers stored as text
            "amount": [f"{v:,.2f}" for v in rng.uniform(0, 5000, n)],  # "1,234.50"
            "city": city,
            "signup": pd.date_range("2021-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
            "code": [f"0{rng.integers(1000, 1010)}" for _ in range(n)],  # leading zeros
            "comment": [f"free text number {i}" for i in range(n)],
            "constant": 7,
            "empty": np.nan,
            "label": np.where(signal > 0.5, "yes", "no"),
        }
    )
    frame.loc[rng.random(n) < 0.1, "x_text"] = None
    return frame


# --- Column roles and types -----------------------------------------------------------------


def test_infer_plan_assigns_roles_and_explains_drops() -> None:
    plan = infer_plan(messy_frame(), "label")
    assert plan.roles == {
        "x_text": Role.NUMERIC,
        "amount": Role.NUMERIC,
        "city": Role.CATEGORICAL,
        "signup": Role.DATETIME,
        "code": Role.CATEGORICAL,
    }
    assert plan.dropped == {
        "customer_id": "row identifier",
        "comment": "high-cardinality text (identifier or free text)",
        "constant": "constant",
        "empty": "all values missing",
    }


def test_read_table_keeps_leading_zero_codes(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("zip,age,score\n02103,41,1.5\n10001,35,\n")
    frame = read_table(path)
    assert list(frame["zip"]) == ["02103", "10001"]
    assert frame["age"].dtype == np.int64
    assert frame["score"].dtype == float


# --- Task detection and row cleaning --------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "task"),
    [
        (["a", "b", "a"], Task.BINARY),
        ([True, False, True], Task.BINARY),
        ([1, 2, 3, 1], Task.MULTICLASS),
        ([0.5, 1.7, 2.2], Task.REGRESSION),
        (list(range(50)), Task.REGRESSION),
    ],
)
def test_infer_task(values: list[object], task: Task) -> None:
    assert infer_task(pd.Series(values, name="y")) is task


def test_infer_task_rejects_single_value_and_accepts_override() -> None:
    with pytest.raises(DataError, match="distinct"):
        infer_task(pd.Series(["a", "a"], name="y"))
    assert infer_task(pd.Series([1, 2, 3]), "regression") is Task.REGRESSION


def test_clean_rows_reports_what_it_removed() -> None:
    frame = pd.DataFrame(
        {"x": [*range(20), 0, 99, 98], "y": [*(["a", "b"] * 10), "a", None, "rare"]}
    )
    frame.loc[20, "x"] = 0  # exact duplicate of row 0
    cleaned, report = clean_rows(frame, "y", Task.BINARY)
    assert report.describe() == {
        "rows_in": 23,
        "rows_out": 20,
        "missing_label": 1,
        "duplicates": 1,
        "rare_classes_dropped": {"rare": 1},
    }
    assert len(cleaned) == 20


def test_clean_rows_explains_unusable_data() -> None:
    with pytest.raises(DataError, match="not found"):
        clean_rows(pd.DataFrame({"x": [1]}), "y", Task.BINARY)
    with pytest.raises(DataError, match="usable rows"):
        clean_rows(pd.DataFrame({"x": range(6), "y": ["a", "b"] * 3}), "y", Task.BINARY)


def test_split_puts_every_class_on_both_sides_of_small_data() -> None:
    frame = pd.DataFrame({"x": range(30), "y": [c for c in "abcdefghij" for _ in range(3)]})
    train, test = split(frame, "y", Task.MULTICLASS, test_size=0.1, seed=0)
    assert set(train["y"]) == set(test["y"]) == set("abcdefghij")


# --- Training on any data -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("loader", "metric", "minimum"),
    [
        (datasets.load_breast_cancer, "accuracy", 0.93),
        (datasets.load_wine, "accuracy", 0.9),
        (datasets.load_iris, "accuracy", 0.9),
        (datasets.load_diabetes, "r2", 0.35),
    ],
)
def test_reference_datasets_reach_known_accuracy(
    loader: object, metric: str, minimum: float
) -> None:
    frame = loader(as_frame=True).frame  # type: ignore[operator]
    task = infer_task(frame["target"])
    train, test = split(frame, "target", task, 0.25, seed=0)
    selection = select_model(train, "target", task=task, settings=QUICK)
    assert evaluate(selection.model, test, "target", task)[metric] >= minimum
    assert selection.chosen.gap <= QUICK.max_gap


def test_nonlinear_data_picks_trees_and_beats_linear() -> None:
    X, y = datasets.make_moons(1500, noise=0.25, random_state=0)
    frame = pd.DataFrame(X, columns=["a", "b"]).assign(label=np.where(y == 1, "yes", "no"))
    train, test = split(frame, "label", Task.BINARY, 0.25, seed=0)
    chosen = select_model(train, "label", settings=QUICK)
    linear = select_model(train, "label", settings=SearchSettings(families=(Family.LINEAR,)))
    assert chosen.chosen.family in {Family.BOOSTING, Family.FOREST}
    chosen_acc = evaluate(chosen.model, test, "label", Task.BINARY)["accuracy"]
    linear_acc = evaluate(linear.model, test, "label", Task.BINARY)["accuracy"]
    assert chosen_acc > linear_acc + 0.03


def test_messy_training_data_and_broken_batches_do_not_crash() -> None:
    frame = messy_frame()
    selection = select_model(frame, "label", settings=QUICK)
    assert evaluate(selection.model, frame, "label", Task.BINARY)["accuracy"] > 0.75

    batch = messy_frame(50, seed=1).drop(columns=["city", "label"])  # a feature is missing
    batch["x_text"] = "not a number"  # unparseable values
    batch["code"] = "09999"  # a category never seen in training
    batch["surprise"] = 1  # an extra column
    proba = selection.model.predict_proba(batch)
    assert proba.shape == (50, 2)
    assert np.isfinite(proba).all()


def test_tiny_data_uses_only_the_linear_family_and_fewer_folds() -> None:
    frame = pd.DataFrame({"x": np.arange(12.0), "y": ["a"] * 9 + ["b"] * 3})
    selection = select_model(frame, "y", settings=QUICK)
    assert selection.chosen.family is Family.LINEAR
    assert selection.chosen.folds == 3  # the smallest class has 3 rows


def test_no_usable_features_is_a_clear_error() -> None:
    frame = pd.DataFrame({"constant": [1] * 20, "y": ["a", "b"] * 10})
    with pytest.raises(DataError, match="no usable feature columns"):
        select_model(frame, "y", settings=QUICK)


def test_overfit_model_records_its_gap() -> None:
    frame = messy_frame(300)
    selection = fit_fixed(frame, "label", Family.FOREST, settings=QUICK)
    assert selection.chosen.gap > 0.1
    assert "fixed by configuration" in selection.reason


def test_selected_model_pickles_and_passes_the_safety_scan(tmp_path: Path) -> None:
    selection = select_model(messy_frame(200), "label", settings=QUICK)
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(selection.model))
    loaded = safe_load_pickle(path)
    assert list(loaded.classes_) == ["no", "yes"]


# --- Selection rules ------------------------------------------------------------------------


def _trial(family: Family, score: float, gap: float, std: float = 0.02) -> Trial:
    return Trial(family, {}, score, std, fit_train=0.8 + gap, fit_validation=0.8, folds=4)


def test_choose_prefers_the_simplest_family_within_one_standard_error() -> None:
    trials = [_trial(Family.LINEAR, -0.400, 0.01), _trial(Family.BOOSTING, -0.395, 0.02)]
    chosen, reason = choose(trials, max_gap=0.05)
    assert chosen.family is Family.LINEAR  # 0.005 better is within one SE (0.01)
    assert "one standard error" in reason


def test_choose_takes_a_clearly_better_complex_model() -> None:
    trials = [_trial(Family.LINEAR, -0.40, 0.01), _trial(Family.BOOSTING, -0.30, 0.02)]
    assert choose(trials, max_gap=0.05)[0].family is Family.BOOSTING


def test_choose_sets_aside_overfit_candidates() -> None:
    trials = [_trial(Family.LINEAR, -0.40, 0.01), _trial(Family.FOREST, -0.20, 0.20)]
    assert choose(trials, max_gap=0.05)[0].family is Family.LINEAR
    only_overfit = [_trial(Family.FOREST, -0.2, 0.2), _trial(Family.BOOSTING, -0.3, 0.1)]
    chosen, reason = choose(only_overfit, max_gap=0.05)
    assert chosen.family is Family.BOOSTING  # least overfit
    assert "least overfit" in reason


# --- Evaluation -----------------------------------------------------------------------------


def test_integer_classes_line_up_with_probability_columns() -> None:
    # Classes 2 and 10 sort differently as numbers and as text; metrics must not mix them up.
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    frame = pd.DataFrame({"x": x, "y": np.where(x > 0, 10, 2)})
    model = select_model(frame, "y", settings=QUICK).model
    metrics = evaluate(model, frame, "y", Task.BINARY)
    assert metrics["accuracy"] > 0.95
    assert metrics["roc_auc"] > 0.95
    as_floats = evaluate(model, frame.astype({"y": float}), "y", Task.BINARY)  # read with NaN
    assert as_floats["accuracy"] == metrics["accuracy"]


def test_unknown_labels_are_counted_not_fatal() -> None:
    frame = messy_frame(200)
    model = select_model(frame, "label", settings=QUICK).model
    batch = frame.assign(label=["maybe"] * 10 + list(frame["label"][10:]))
    metrics = evaluate(model, batch, "label", Task.BINARY)
    assert metrics["unknown_label_rows"] == 10


def test_regression_metrics_include_a_baseline() -> None:
    frame = datasets.load_diabetes(as_frame=True).frame
    model = Ridge().fit(frame.drop(columns=["target"]), frame["target"])
    metrics = evaluate(model, frame, "target", Task.REGRESSION)
    assert metrics["rmse"] < metrics["baseline_rmse"]
    assert 0 < metrics["r2"] < 1


def test_leakage_audit_supports_regressors() -> None:
    frame = datasets.load_diabetes(as_frame=True).frame
    train, test = frame.iloc[:300], frame.iloc[300:]
    model = Ridge().fit(train.drop(columns=["target"]), train["target"])
    result = loss_threshold_attack(model, train, test, "target")
    assert result.gap_metric == "r2"
    assert 0.3 < result.mia_auc < 0.7
    classifier = LogisticRegression().fit(train.drop(columns=["target"]), train["target"] > 140)
    assert classifier.predict_proba(test.drop(columns=["target"])).shape[1] == 2


# --- Privacy preprocessing ------------------------------------------------------------------


def test_identifiers_are_detected_in_any_table() -> None:
    fake = Faker()
    Faker.seed(0)
    frame = pd.DataFrame(
        {
            "contact": [fake.email() for _ in range(60)],
            "full_name": [fake.name() for _ in range(60)],
            "score": range(60),
            "label": ["a", "b"] * 30,
        }
    )
    assert detect_identifiers(frame, keep=["label"]) == ["contact", "full_name"]


def test_generalization_learns_rules_once_and_applies_them_to_batches() -> None:
    frame = pd.DataFrame(
        {
            "age": np.arange(18, 80),
            "zip": [f"0{2100 + i % 7}" for i in range(62)],
            "flag": [0, 1] * 31,  # two values: left alone
        }
    )
    rules = fit_generalization(frame, ["age", "zip", "flag"])
    assert set(rules) == {"age", "zip"}
    batch = pd.DataFrame({"age": [19, 45, 90, None], "zip": ["02101", "02199", None, "02100"]})
    out = apply_generalization(batch, rules)
    assert list(out["age"].fillna("?")) == ["<30", "40-49", "70+", "?"]
    assert list(out["zip"].fillna("?")) == ["021**", "021**", "?", "021**"]


def test_narrow_numeric_ranges_get_a_finer_bucket_width() -> None:
    rules = fit_generalization(pd.DataFrame({"score": np.linspace(0, 1, 50)}), ["score"])
    assert rules["score"]["width"] == 0.2
    labels = {range_label(v, rules["score"]) for v in np.linspace(0, 1, 50)}
    assert labels == {"<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8+"}


# --- Ready-made stages ----------------------------------------------------------------------


def test_stages_run_in_sequence_on_any_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from model_passport.ml import stages

    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    messy_frame(300).to_csv("data/raw.csv", index=False)
    common = {"label": "label"}
    for name, extra in (("preprocess", {}), ("train", {"search": "quick"}), ("evaluate", {})):
        monkeypatch.setenv("PASSPORT_PARAMS", json.dumps({**common, **extra}))
        stages.run(name)
    output = capsys.readouterr().out
    assert "binary-classification" in output
    assert "chose" in output
    assert "unused columns: customer_id (row identifier)" in output
    assert "test accuracy" in output
    manifest = json.loads(Path("data/preprocess.json").read_text())
    assert manifest["task"] == "binary-classification"
    info = json.loads(Path("models/model_info.json").read_text())
    assert info["output_schema"] == {"label": ["no", "yes"]}
    assert json.loads(Path("models/metrics.json").read_text())["test"]["accuracy"] > 0.7

    monkeypatch.setenv("PASSPORT_PARAMS", json.dumps({**common, "model": "overfit"}))
    stages.run("train")
    assert "fixed by configuration" in capsys.readouterr().out


def test_stages_exit_with_a_readable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from model_passport.ml import stages

    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    pd.DataFrame({"x": range(30), "y": ["a"] * 30}).to_csv("data/raw.csv", index=False)
    monkeypatch.setenv("PASSPORT_PARAMS", json.dumps({"label": "y"}))
    with pytest.raises(SystemExit, match="preprocess: label 'y' has 1 distinct value"):
        stages.run("preprocess")
    monkeypatch.setenv("PASSPORT_PARAMS", json.dumps({"label": "missing"}))
    with pytest.raises(SystemExit, match="label column 'missing' not in"):
        stages.run("preprocess")
    monkeypatch.delenv("PASSPORT_PARAMS")
    with pytest.raises(SystemExit, match="set `label`"):
        stages.run("train")


# --- Adaptive k-anonymity -------------------------------------------------------------------


def _people(n: int, seed: int = 0) -> pd.DataFrame:
    """Four quasi-identifiers; only age and marital status predict the label."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n)
    prefixes = ["021", "100", "303", "606", "941"]
    marital = rng.choice(
        ["married", "single", "divorced", "widowed"], n, p=[0.45, 0.35, 0.15, 0.05]
    )
    signal = 0.06 * (age - 45) + 1.2 * (marital == "married") + rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "age": age,
            "gender": rng.choice(["F", "M"], n),
            "zip": [f"{p}{rng.integers(10, 99)}" for p in rng.choice(prefixes, n)],
            "marital_status": marital,
            "hours": rng.normal(40, 8, n).round(),
            "label": np.where(signal > 0.5, "yes", "no"),
        }
    )


def test_generalization_coarsens_identifying_columns_before_predictive_ones() -> None:
    from model_passport.ml.privacy import at_risk

    frame = _people(800)
    qi = ["age", "gender", "zip", "marital_status"]
    rules = fit_generalization(frame, qi, target_k=20, label=frame["label"])
    out = apply_generalization(frame, rules)
    assert at_risk(out, qi, 20) <= 0.01 * len(frame)
    # zip carries no information about the label, so it gives way; age predicts the label
    # and keeps several ranges.
    assert out["zip"].nunique() == 1
    assert rules["age"]["kind"] == "range"
    assert out["age"].nunique() >= 2


def test_preprocess_reaches_k_anonymity_in_both_files_on_small_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from model_passport.ml import stages
    from model_passport.scanners.reid_risk import k_anonymity

    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    _people(600).to_csv("data/raw.csv", index=False)
    monkeypatch.setenv("PASSPORT_PARAMS", json.dumps({"label": "label"}))
    stages.run("preprocess")
    manifest = json.loads(Path("data/preprocess.json").read_text())
    qi = manifest["quasi_identifiers"]
    assert set(qi) == {"age", "gender", "zip", "marital_status"}
    removed = sum(manifest["k_anonymity"]["suppressed"].values())
    assert removed <= 0.05 * 600  # anonymity comes from coarsening, not from deleting people
    for name in ("train", "test"):
        part = read_table(Path(f"data/{name}.csv"))
        assert k_anonymity(part, qi) >= 5


def test_generalization_without_a_target_keeps_the_initial_buckets() -> None:
    frame = _people(300)
    rules = fit_generalization(frame, ["age", "zip", "gender"])
    assert rules["age"]["width"] == 10
    assert rules["zip"] == {"kind": "prefix", "keep": 3}
    assert "gender" not in rules


def test_suppress_small_groups_and_pooling() -> None:
    from model_passport.ml.privacy import pool_label, suppress_small_groups

    frame = pd.DataFrame({"a": ["x"] * 6 + ["y"] * 2, "b": [1] * 8})
    kept, removed = suppress_small_groups(frame, ["a", "b"], k=5)
    assert (len(kept), removed) == (6, 2)
    assert pool_label("rare", {"common"}) == "other"
    assert pool_label("common", {"common"}) == "common"
