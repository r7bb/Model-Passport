"""Assemble, hash, and sign a passport from ``passport.yaml`` build inputs."""

from __future__ import annotations

import json
from pathlib import Path

from model_passport.core import identity
from model_passport.core.config import ProjectConfig, load_config
from model_passport.core.schema import (
    ArtifactKind,
    ArtifactRef,
    DatasetInfo,
    Identity,
    ModelInfo,
    Passport,
)


class BuildError(Exception):
    """Raised when build inputs are missing or invalid."""


def _relative_path(root: Path, path: Path) -> str:
    """POSIX path relative to ``root``; absolute if the file lives outside the project."""
    resolved = (root / path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _hash_artifact(root: Path, path: Path, kind: ArtifactKind) -> ArtifactRef:
    full = (root / path).resolve()
    if not full.is_file():
        raise BuildError(f"{kind.value} artifact not found: {path}")
    return ArtifactRef(
        path=_relative_path(root, path),
        sha256=identity.sha256_file(full),
        size_bytes=full.stat().st_size,
        kind=kind,
    )


def build_passport(config_path: Path, config: ProjectConfig | None = None) -> Passport:
    """Build and sign a passport. Paths in the config are resolved against its directory."""
    root = config_path.parent.resolve()
    config = config or load_config(config_path)
    inputs = config.build

    artifacts: dict[str, ArtifactRef] = {}

    def add(path: Path, kind: ArtifactKind) -> ArtifactRef:
        ref = _hash_artifact(root, path, kind)
        return artifacts.setdefault(ref.path, ref)

    add(config_path.resolve(), ArtifactKind.CONFIG)
    model_ref = add(inputs.model.path, ArtifactKind.MODEL)
    datasets = []
    for ds in inputs.datasets:
        ref = add(ds.path, ArtifactKind.DATASET)
        datasets.append(
            DatasetInfo(
                name=ds.name or Path(ds.path).stem,
                source=ds.source,
                license=ds.license,
                sha256=ref.sha256,
                split_role=ds.split_role,
            )
        )
    for extra in inputs.artifacts:
        add(extra.path, extra.kind)

    private_key_path = root / config.signing.private_key
    if not private_key_path.is_file():
        raise BuildError(
            f"signing key not found: {config.signing.private_key} (run `passport init`)"
        )
    private_key = identity.load_private_key(private_key_path)

    manifest = sorted(artifacts.values(), key=lambda a: a.path)
    passport = Passport(
        identity=Identity(
            model_name=config.project.name,
            version=config.project.version,
            merkle_root=identity.merkle_root(a.sha256 for a in manifest),
            public_key_fingerprint=identity.public_key_fingerprint(private_key.public_key()),
        ),
        artifacts=manifest,
        model=ModelInfo(
            framework=inputs.model.framework,
            algorithm=inputs.model.algorithm,
            task_type=inputs.model.task_type,
            hyperparameters=inputs.model.hyperparameters,
            artifact_uri=model_ref.path,
            artifact_sha256=model_ref.sha256,
            input_schema=inputs.model.input_schema,
            output_schema=inputs.model.output_schema,
        ),
        datasets=datasets,
        metrics=inputs.metrics,
        declared=config.declared,
        lineage_links=inputs.lineage_links,
    )
    return sign_passport(passport, private_key)


def sign_passport(passport: Passport, private_key: identity.Ed25519PrivateKey) -> Passport:
    payload = identity.signing_payload(passport.model_dump(mode="json"))
    signed = passport.model_copy(deep=True)
    signed.identity.signature = identity.sign(private_key, payload)
    return signed


def write_passport(passport: Passport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(passport.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
