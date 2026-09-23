from __future__ import annotations

from pathlib import Path

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
