from __future__ import annotations

import json
import os
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from model_passport.auditors import artifact
from model_passport.auditors.artifact import (
    UnsafeArtifactError,
    audit_dependencies,
    safe_load_pickle,
    scan_model_file,
)
from model_passport.auditors.leakage import (
    LeakageAuditError,
    generalization_gap,
    loss_threshold_attack,
    tpr_at_fpr,
)
from model_passport.core.schema import DependencyAudit, Severity


class _Exploit:
    """Pickles to a call of os.system. Never unpickled by these tests."""

    def __reduce__(self) -> tuple:
        return (os.system, ("echo pwned",))


def _data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    noisy = (x[:, 0] + rng.normal(scale=2.0, size=n)) > 0
    return pd.DataFrame(x, columns=[f"f{i}" for i in range(5)]).assign(y=noisy.astype(int))


# --- Leakage -------------------------------------------------------------------------------


def test_overfit_model_leaks_more_than_regularized() -> None:
    train, test = _data(600, 1), _data(600, 2)
    X, y = train.drop(columns="y"), train["y"]
    overfit = DecisionTreeClassifier(random_state=0).fit(X, y)
    regular = LogisticRegression(C=0.1).fit(X, y)

    leaky = loss_threshold_attack(overfit, train, test, "y")
    safe = loss_threshold_attack(regular, train, test, "y")
    assert leaky.mia_auc > 0.7
    assert abs(safe.mia_auc - 0.5) < 0.08
    assert leaky.mia_auc > safe.mia_auc
    assert leaky.generalization_gap > safe.generalization_gap
    assert 0.0 <= leaky.tpr_at_low_fpr <= 1.0
    assert (leaky.members, leaky.nonmembers) == (600, 600)


def test_tpr_at_fpr() -> None:
    labels = np.array([1, 1, 0, 0])
    assert tpr_at_fpr(labels, np.array([0.9, 0.8, 0.1, 0.2]), 0.01) == 1.0
    assert tpr_at_fpr(labels, np.array([0.1, 0.2, 0.9, 0.8]), 0.01) == 0.0


def test_log_loss_gap_and_errors() -> None:
    train, test = _data(200, 3), _data(200, 4)
    model = DecisionTreeClassifier(random_state=0).fit(train.drop(columns="y"), train["y"])
    assert generalization_gap(model, train, test, "y", "log_loss") > 0
    with pytest.raises(LeakageAuditError, match="unsupported gap metric"):
        generalization_gap(model, train, test, "y", "f7")
    with pytest.raises(LeakageAuditError, match="label column"):
        loss_threshold_attack(model, train, test, "missing")
    with pytest.raises(LeakageAuditError, match="predict_proba"):
        loss_threshold_attack(object(), train, test, "y")


# --- Artifact scanning ---------------------------------------------------------------------


def test_scan_flags_dangerous_pickle_and_refuses_to_load(tmp_path: Path) -> None:
    evil = tmp_path / "model.pkl"
    evil.write_bytes(pickle.dumps(_Exploit()))
    findings = {f.category: f for f in scan_model_file(evil)}
    assert findings["UNSAFE_PICKLE"].severity is Severity.CRITICAL
    assert any("system" in g for g in findings["UNSAFE_PICKLE"].details["dangerous_globals"])
    with pytest.raises(UnsafeArtifactError):
        safe_load_pickle(evil)


def test_scan_benign_sklearn_pickle(tmp_path: Path) -> None:
    train = _data(50, 5)
    model = LogisticRegression().fit(train.drop(columns="y"), train["y"])
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(model))
    categories = {f.category for f in scan_model_file(path)}
    assert categories == {"RAW_PICKLE_FORMAT"}
    assert isinstance(safe_load_pickle(path), LogisticRegression)


def test_scan_safe_and_unknown_formats(tmp_path: Path) -> None:
    (tmp_path / "m.safetensors").write_bytes(b"{}")
    (tmp_path / "m.xyz").write_bytes(b"?")
    assert scan_model_file(tmp_path / "m.safetensors")[0].category == "SAFE_FORMAT"
    assert scan_model_file(tmp_path / "m.xyz")[0].category == "UNKNOWN_FORMAT"


def test_corrupt_pickle_is_blocking(tmp_path: Path) -> None:
    path = tmp_path / "broken.pkl"
    path.write_bytes(b"\x80\x05not a pickle")
    categories = {f.category for f in scan_model_file(path)}
    assert categories & {"SCAN_ERROR", "RAW_PICKLE_FORMAT"}
    if "SCAN_ERROR" in categories:
        with pytest.raises(UnsafeArtifactError):
            safe_load_pickle(path)


# --- Dependency audit ----------------------------------------------------------------------

PIP_AUDIT_OUTPUT = {
    "dependencies": [
        {
            "name": "jinja2",
            "version": "2.10",
            "vulns": [
                {"id": "PYSEC-2019-217", "aliases": ["GHSA-462w-v97r-4m45", "CVE-2019-10906"],
                 "fix_versions": ["2.10.1"]},
                {"id": "PYSEC-2019-217", "aliases": ["GHSA-462w-v97r-4m45", "CVE-2019-10906"],
                 "fix_versions": ["2.10.1"]},
                {"id": "PYSEC-2021-66", "aliases": ["CVE-2020-28493"], "fix_versions": []},
            ],
        },
        {"name": "numpy", "version": "2.0.0", "vulns": []},
    ]
}  # fmt: skip


def test_audit_dependencies_dedupes_and_maps_severity(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        req = Path(cmd[cmd.index("-r") + 1]).read_text()
        assert "jinja2==2.10" in req and "model-passport" not in req
        return subprocess.CompletedProcess(cmd, 1, json.dumps(PIP_AUDIT_OUTPUT), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        artifact,
        "fetch_severity",
        lambda vid: Severity.CRITICAL if vid.startswith("GHSA") else None,
    )
    status, _, findings = audit_dependencies(
        {"jinja2": "2.10", "numpy": "2.0.0", "model-passport": "0.1.0"}
    )
    assert status is DependencyAudit.OK
    by_id = {f.details["id"]: f for f in findings}
    assert set(by_id) == {"PYSEC-2019-217", "PYSEC-2021-66"}
    assert by_id["PYSEC-2019-217"].category == "CVE-2019-10906"
    assert by_id["PYSEC-2019-217"].severity is Severity.CRITICAL
    assert by_id["PYSEC-2021-66"].details["severity_source"] == "unknown"


def test_audit_dependencies_reports_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **_: subprocess.CompletedProcess(cmd, 2, "", "network unreachable"),
    )
    status, message, findings = audit_dependencies({"numpy": "2.0.0"})
    assert status is DependencyAudit.ERROR and "network unreachable" in message and not findings
