"""Assemble, hash, and sign a passport from ``passport.yaml`` build inputs."""

from __future__ import annotations

import json
from pathlib import Path

from model_passport.core import identity
from model_passport.core.assess import (
    AssessmentError,
    assess_models,
    assess_privacy,
    scan_secrets,
)
from model_passport.core.capture import (
    RunRecord,
    capture_environment,
    load_run_record,
    stages_fingerprint,
)
from model_passport.core.config import ProjectConfig, load_config
from model_passport.core.revision import compute_revision
from model_passport.core.schema import (
    ArtifactKind,
    ArtifactRef,
    DatasetInfo,
    Identity,
    ModelInfo,
    Passport,
    RunInfo,
    SecurityReport,
)
from model_passport.policy.engine import PolicyError, evaluate, load_policy


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


class _Manifest:
    """Collects hashed artifacts, keyed by path; the first declared kind wins."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.items: dict[str, ArtifactRef] = {}

    def add(self, path: Path | str, kind: ArtifactKind) -> ArtifactRef:
        ref = _hash_artifact(self.root, Path(path), kind)
        return self.items.setdefault(ref.path, ref)

    def sorted(self) -> list[ArtifactRef]:
        return sorted(self.items.values(), key=lambda a: a.path)


def _load_current_run(root: Path, config: ProjectConfig) -> RunRecord | None:
    if not config.stages:
        return None
    run = load_run_record(root)
    if run is None:
        raise BuildError("stages are declared but no run record exists (run `passport run`)")
    if run.stages_sha256 != stages_fingerprint(config.stages):
        raise BuildError("stages in passport.yaml changed since the last `passport run`")
    return run


def _add_run_artifacts(manifest: _Manifest, run: RunRecord) -> None:
    """Add stage scripts, inputs, and outputs, failing if any changed since the run."""
    for stage in run.stages:
        recorded = [(stage.script_path, stage.script_sha256, ArtifactKind.SCRIPT)] + [
            (ref.path, ref.sha256, ref.kind) for ref in (*stage.inputs, *stage.outputs)
        ]
        for path, sha256, kind in recorded:
            if manifest.add(path, kind).sha256 != sha256:
                raise BuildError(
                    f"{path} changed since `passport run` (stage {stage.name!r}); rerun"
                )


def build_passport(
    config_path: Path,
    config: ProjectConfig | None = None,
    previous: Passport | None = None,
    reason: str | None = None,
) -> Passport:
    """Build and sign a passport. Paths in the config are resolved against its directory.

    With ``previous``, the new passport records what changed since that version (new data,
    retraining) and links to it, forming a verifiable version history.
    """
    root = config_path.parent.resolve()
    config = config or load_config(config_path)
    inputs = config.build
    run = _load_current_run(root, config)

    manifest = _Manifest(root)
    manifest.add(config_path.resolve(), ArtifactKind.CONFIG)
    model_ref = manifest.add(inputs.model.path, ArtifactKind.MODEL)
    datasets: list[tuple[str, DatasetInfo]] = []
    for ds in inputs.datasets:
        ref = manifest.add(ds.path, ArtifactKind.DATASET)
        info = DatasetInfo(
            name=ds.name or Path(ds.path).stem,
            source=ds.source,
            license=ds.license,
            sha256=ref.sha256,
            split_role=ds.split_role,
        )
        datasets.append((ref.path, info))
    for extra in inputs.artifacts:
        manifest.add(extra.path, extra.kind)
    if run is not None:
        _add_run_artifacts(manifest, run)

    policy = None
    if config.policy is not None:
        policy_ref = manifest.add(config.policy, ArtifactKind.CONFIG)
        try:
            policy, policy_sha256 = load_policy(root / policy_ref.path)
        except PolicyError as exc:
            raise BuildError(str(exc)) from exc

    model_fields = inputs.model.model_dump(exclude={"path", "metadata"}, exclude_defaults=True)
    if inputs.model.metadata is not None:
        manifest.add(inputs.model.metadata, ArtifactKind.OTHER)
        declared_meta = json.loads((root / inputs.model.metadata).read_text(encoding="utf-8"))
        model_fields = {**declared_meta, **model_fields}

    private_key_path = root / config.signing.private_key
    if not private_key_path.is_file():
        raise BuildError(
            f"signing key not found: {config.signing.private_key} (run `passport init`)"
        )
    private_key = identity.load_private_key(private_key_path)

    artifacts = manifest.sorted()
    environment = run.environment if run else capture_environment()
    security_report = SecurityReport()
    try:
        privacy_report = assess_privacy(root, config.privacy, datasets)
        privacy_report.leakage = assess_models(
            root,
            config.audit,
            model_ref.path,
            artifacts,
            datasets,
            environment,
            security_report,
            input_schema=model_fields.get("input_schema"),
        )
    except AssessmentError as exc:
        raise BuildError(str(exc)) from exc
    if config.privacy.scan_secrets:
        scanned, secret_findings = scan_secrets(root, artifacts)
        security_report.files_scanned_for_secrets = scanned
        security_report.secret_findings = secret_findings

    metrics = {split: dict(values) for split, values in (run.metrics if run else {}).items()}
    for split, values in inputs.metrics.items():
        metrics.setdefault(split, {}).update(values)

    passport = Passport(
        identity=Identity(
            model_name=config.project.name,
            version=config.project.version,
            merkle_root=identity.merkle_root(a.sha256 for a in artifacts),
            public_key_fingerprint=identity.public_key_fingerprint(private_key.public_key()),
        ),
        artifacts=artifacts,
        model=ModelInfo(
            **model_fields, artifact_uri=model_ref.path, artifact_sha256=model_ref.sha256
        ),
        datasets=[info for _, info in datasets],
        pipeline=run.stages if run else [],
        run=(
            RunInfo(
                run_id=run.run_id,
                started_at=run.started_at,
                ended_at=run.ended_at,
                overrides=run.overrides,
                mlflow_run_id=run.mlflow_run_id,
            )
            if run
            else None
        ),
        environment=environment,
        metrics=metrics,
        privacy_report=privacy_report,
        security_report=security_report,
        declared=config.declared,
        lineage_links=inputs.lineage_links,
    )
    if previous is not None:
        passport.revision = compute_revision(previous, passport, reason)
    if policy is not None:
        passport.policy = evaluate(policy, policy_sha256, passport)
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
