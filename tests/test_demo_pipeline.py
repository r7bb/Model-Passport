"""End-to-end: the real demo pipeline in a temporary copy of the repo layout."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("faker")

from model_passport.core import identity  # noqa: E402
from model_passport.core.builder import build_passport, write_passport  # noqa: E402
from model_passport.core.capture import run_pipeline, save_run_record  # noqa: E402
from model_passport.core.config import load_config  # noqa: E402
from model_passport.core.verifier import verify_passport  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("demo")
    shutil.copytree(REPO / "demo", root / "demo")
    shutil.copy(REPO / "passport.yaml", root / "passport.yaml")
    shutil.copy(REPO / "policy.yaml", root / "policy.yaml")
    subprocess.run(
        [sys.executable, "demo/make_dataset.py", "--rows", "4000"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    identity.save_keypair(
        identity.generate_keypair(),
        root / ".passport/signing_key.pem",
        root / ".passport/signing_key.pub",
    )
    return root


UNSAFE = {
    "preprocess": {"drop_identifiers": False, "generalize": False},
    "train": {"model": "overfit"},
}
PRIVACY_RULES = {"pii_columns_max", "min_k_anonymity", "unique_record_fraction_max"}


def run_demo(root: Path, overrides: dict | None = None) -> Path:
    config = load_config(root / "passport.yaml")
    save_run_record(root, run_pipeline(root, config, overrides))
    out = root / "passport.json"
    write_passport(build_passport(root / "passport.yaml", config), out)
    return out


def _rule_results(out: Path) -> dict[str, str]:
    import json

    policy = json.loads(out.read_text())["policy"]
    return {r["name"]: r["result"] for r in policy["rules"]}


def test_injected_pii_fails_and_cleaned_data_passes(demo_project: Path) -> None:
    unsafe = _rule_results(run_demo(demo_project, UNSAFE))
    assert {unsafe[r] for r in PRIVACY_RULES} == {"fail"}

    safe = _rule_results(run_demo(demo_project))
    assert {safe[r] for r in PRIVACY_RULES} == {"pass"}
    assert safe["secrets_found_max"] == "pass"


def test_demo_passport_has_three_stages(demo_project: Path) -> None:
    import json

    out = run_demo(demo_project)
    passport = json.loads(out.read_text())
    stages = passport["pipeline"]
    assert [s["name"] for s in stages] == ["preprocess", "train", "evaluate"]
    assert stages[0]["parameters"]["drop_identifiers"] is True
    assert [i["path"] for i in stages[0]["inputs"]] == ["data/raw.csv"]
    assert [o["path"] for o in stages[1]["outputs"]] == [
        "models/model.pkl",
        "models/model_info.json",
    ]
    assert passport["model"]["algorithm"] == "LogisticRegression"
    assert set(passport["metrics"]) == {"train", "test"}
    assert verify_passport(out, demo_project / ".passport/signing_key.pub", demo_project).ok
