"""End-to-end: the real demo pipeline in a temporary copy of the repo layout."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("faker")

from model_passport.core import identity  # noqa: E402
from model_passport.core.builder import build_passport, write_passport  # noqa: E402
from model_passport.core.capture import run_pipeline, save_run_record  # noqa: E402
from model_passport.core.config import load_config  # noqa: E402
from model_passport.core.verifier import verify_passport  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
UNSAFE = {
    "preprocess": {"drop_identifiers": False, "generalize": False},
    "train": {"model": "overfit"},
}
PRIVACY_RULES = {"pii_columns_max", "min_k_anonymity", "unique_record_fraction_max"}


def _python(root: Path, *args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=root, check=True, capture_output=True)


@pytest.fixture(scope="module")
def demo_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("demo")
    shutil.copytree(REPO / "demo", root / "demo")
    shutil.copy(REPO / "passport.yaml", root / "passport.yaml")
    shutil.copy(REPO / "policy.yaml", root / "policy.yaml")
    _python(root, "demo/make_dataset.py", "--rows", "4000")
    identity.save_keypair(
        identity.generate_keypair(),
        root / ".passport/signing_key.pem",
        root / ".passport/signing_key.pub",
    )
    return root


def run_demo(root: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_config(root / "passport.yaml")
    save_run_record(root, run_pipeline(root, config, overrides))
    write_passport(build_passport(root / "passport.yaml", config), root / "passport.json")
    return json.loads((root / "passport.json").read_text())


def _rules(passport: dict[str, Any]) -> dict[str, str]:
    return {r["name"]: r["result"] for r in passport["policy"]["rules"]}


def test_demo_passport_records_pipeline_and_metrics(demo_project: Path) -> None:
    passport = run_demo(demo_project)
    stages = passport["pipeline"]
    assert [s["name"] for s in stages] == ["preprocess", "train", "evaluate"]
    assert stages[0]["parameters"]["drop_identifiers"] is True
    assert [i["path"] for i in stages[0]["inputs"]] == ["data/raw.csv"]
    assert [o["path"] for o in stages[1]["outputs"]] == [
        "models/model.pkl",
        "models/model_info.json",
        "models/cv_metrics.json",
    ]
    model = passport["model"]
    assert model["algorithm"] == "LogisticRegression"
    assert model["hyperparameters"]["C"] in model["hyperparameters"]["C_grid"]
    assert set(passport["metrics"]) == {"train", "test", "cv"}
    test = passport["metrics"]["test"]
    assert test["accuracy"] > test["baseline_accuracy"]  # better than majority class
    assert {"roc_auc", "log_loss", "brier", "balanced_accuracy"} <= set(test)
    assert 0.5 < passport["metrics"]["cv"]["roc_auc_mean"] <= 1.0
    assert verify_passport(
        demo_project / "passport.json", demo_project / ".passport/signing_key.pub", demo_project
    ).ok


def test_injected_pii_fails_and_cleaned_data_passes(demo_project: Path) -> None:
    unsafe = run_demo(demo_project, UNSAFE)
    assert {_rules(unsafe)[r] for r in PRIVACY_RULES} == {"fail"}
    assert unsafe["policy"]["verdict"] == "fail"

    safe = run_demo(demo_project)
    assert {_rules(safe)[r] for r in PRIVACY_RULES} == {"pass"}
    assert safe["policy"]["verdict"] == "pass"


def test_overfit_model_has_higher_attack_auc(demo_project: Path) -> None:
    overfit = run_demo(demo_project, {"train": {"model": "overfit"}})["privacy_report"]["leakage"]
    regular = run_demo(demo_project)["privacy_report"]["leakage"]
    assert overfit["mia_auc"] > 0.8
    assert regular["mia_auc"] < 0.6
    assert overfit["generalization_gap"] > regular["generalization_gap"]


def test_prepare_batch_uses_recorded_parameters(demo_project: Path) -> None:
    run_demo(demo_project)
    _python(demo_project, "demo/make_dataset.py", "--rows", "50", "--seed", "5",
            "--out", "data/batches/raw.csv")  # fmt: skip
    _python(demo_project, "demo/prepare_batch.py", "data/batches/raw.csv",
            "--out", "data/batches/prepared.csv")  # fmt: skip
    prepared = pd.read_csv(demo_project / "data/batches/prepared.csv", dtype={"zip": str})
    train = pd.read_csv(demo_project / "data/train.csv", dtype={"zip": str})
    assert list(prepared.columns) == list(train.columns)  # identifiers dropped, same layout
    assert prepared["zip"].str.endswith("**").all()
    assert set(prepared["age"]) <= set(train["age"])
