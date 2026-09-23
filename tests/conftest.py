from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from model_passport.core import assess, identity
from model_passport.core.config import CONFIG_FILENAME, CONFIG_TEMPLATE
from model_passport.core.schema import DependencyAudit


@pytest.fixture(autouse=True)
def _no_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(identity.PASSPHRASE_ENV, raising=False)


@pytest.fixture(autouse=True)
def _offline_dependency_audit(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Builds in tests must not call pip-audit or OSV over the network."""
    if "network" in request.keywords:
        return
    monkeypatch.setattr(
        assess,
        "audit_dependencies",
        lambda deps: (DependencyAudit.OK, f"audited {len(deps)} packages (stubbed)", []),
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A small initialized project with a model, two datasets, and a script."""
    root = tmp_path / "proj"
    (root / "models").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "models" / "model.pkl").write_bytes(b"fake-model-bytes")
    (root / "data" / "train.csv").write_text("age,label\n34,1\n51,0\n")
    (root / "data" / "test.csv").write_text("age,label\n29,0\n")
    (root / "train.py").write_text("print('train')\n")

    config = yaml.safe_load(CONFIG_TEMPLATE.format(name="demo-model"))
    config["build"]["model"].update(framework="sklearn", algorithm="LogisticRegression")
    config["build"]["datasets"] = [
        {"path": "data/train.csv", "split_role": "train", "license": "CC-BY-4.0"},
        {"path": "data/test.csv", "split_role": "test"},
    ]
    config["build"]["artifacts"] = [{"path": "train.py", "kind": "script"}]
    config["build"]["metrics"] = {"test": {"accuracy": 0.91}}
    config["declared"]["intended_use"] = "Demo only."
    (root / CONFIG_FILENAME).write_text(yaml.safe_dump(config))

    identity.save_keypair(
        identity.generate_keypair(),
        root / ".passport" / "signing_key.pem",
        root / ".passport" / "signing_key.pub",
    )
    return root


def make_frame(n: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    """Synthetic tabular data; ``shift`` moves the feature distribution (drift)."""
    rng = np.random.default_rng(seed)
    hours = rng.normal(40 + 12 * shift, 8, n)
    group = rng.choice(["a", "b", "c"], n, p=[0.5 - 0.3 * shift, 0.3, 0.2 + 0.3 * shift])
    signal = (hours - 40) / 8 + (group == "a") + rng.normal(0, 1, n)
    return pd.DataFrame({"hours": hours, "group": group, "label": np.where(signal > 0.5, "y", "n")})


@pytest.fixture
def sk_project(project: Path) -> Path:
    """``project`` with a real scikit-learn model, schema metadata, and a signed passport."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    train, test = make_frame(1500, 1), make_frame(500, 2)
    train.to_csv(project / "data/train.csv", index=False)
    test.to_csv(project / "data/test.csv", index=False)
    model = Pipeline(
        [
            ("f", ColumnTransformer([("g", OneHotEncoder(), ["group"])], remainder="passthrough")),
            ("clf", LogisticRegression()),
        ]
    ).fit(train[["hours", "group"]], train["label"])
    (project / "models/model.pkl").write_bytes(pickle.dumps(model))
    accuracy = float((model.predict(test[["hours", "group"]]) == test["label"]).mean())
    info = {
        "algorithm": "LogisticRegression",
        "input_schema": {"hours": "float64", "group": "object"},
        "output_schema": {"label": ["n", "y"]},
    }
    (project / "models/model_info.json").write_text(json.dumps(info))

    config = yaml.safe_load((project / CONFIG_FILENAME).read_text())
    config["build"]["model"] = {"path": "models/model.pkl", "metadata": "models/model_info.json"}
    config["build"]["metrics"] = {"test": {"accuracy": round(accuracy, 4)}}
    config["audit"] = {"label_column": "label"}
    (project / CONFIG_FILENAME).write_text(yaml.safe_dump(config))

    from model_passport.core.builder import build_passport, write_passport

    write_passport(build_passport(project / CONFIG_FILENAME), project / "passport.json")
    return project
