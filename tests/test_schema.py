from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from model_passport.core.config import CONFIG_TEMPLATE, ProjectConfig
from model_passport.core.schema import ArtifactRef, Identity, Passport

SHA = "a" * 64
FP = "sha256:" + "b" * 64


def _passport() -> Passport:
    return Passport(
        identity=Identity(model_name="m", version="1", merkle_root=SHA, public_key_fingerprint=FP),
        artifacts=[ArtifactRef(path="models/m.pkl", sha256=SHA, size_bytes=3, kind="model")],
        metrics={"test": {"accuracy": 1}},
    )


def test_passport_json_roundtrip() -> None:
    passport = _passport()
    data = passport.model_dump(mode="json")
    assert Passport.model_validate(data) == passport
    assert data["metrics"]["test"]["accuracy"] == 1.0
    assert data["identity"]["created_at"].endswith("Z")


def test_rejects_unknown_fields() -> None:
    data = _passport().model_dump(mode="json")
    data["identity"]["surprise"] = 1
    with pytest.raises(ValidationError):
        Passport.model_validate(data)


@pytest.mark.parametrize("bad", ["A" * 64, "a" * 63, "g" * 64])
def test_rejects_malformed_hash(bad: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactRef(path="x", sha256=bad, size_bytes=0)


def test_requires_at_least_one_artifact() -> None:
    data = _passport().model_dump(mode="json")
    data["artifacts"] = []
    with pytest.raises(ValidationError):
        Passport.model_validate(data)


def test_config_template_parses() -> None:
    config = ProjectConfig.model_validate(yaml.safe_load(CONFIG_TEMPLATE.format(name="x")))
    assert config.project.name == "x"
    assert str(config.build.model.path) == "models/model.pkl"
