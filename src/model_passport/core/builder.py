"""Assemble, hash, and sign a passport from ``passport.yaml`` build inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

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
    EntityAuditResult,
    Environment,
    Identity,
    ModelInfo,
    Passport,
    PrivacyReport,
    RunInfo,
    SecurityReport,
)
from model_passport.policy.engine import Policy, PolicyError, evaluate, load_policy


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
    if not full.exists():
        raise BuildError(f"{kind.value} artifact not found: {path}")
    return ArtifactRef(
        path=_relative_path(root, path),
        sha256=identity.sha256_path(full),
        size_bytes=identity.path_size(full),
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
    run = _load_current_run(root, config)

    manifest = _Manifest(root)
    manifest.add(config_path.resolve(), ArtifactKind.CONFIG)
    if config.privacy.entity_audit is not None:
        manifest.add(config.privacy.entity_audit, ArtifactKind.OTHER)
    model_ref = manifest.add(config.build.model.path, ArtifactKind.MODEL)
    datasets = _collect_datasets(manifest, config)
    for extra in config.build.artifacts:
        manifest.add(extra.path, extra.kind)
    if run is not None:
        _add_run_artifacts(manifest, run)
    policy = _load_project_policy(root, config, manifest)
    model_fields = _model_fields(root, config, manifest)
    private_key = _load_signing_key(root, config)

    artifacts = manifest.sorted()
    environment = run.environment if run else capture_environment()
    privacy_report, security_report = _run_checks(
        root, config, model_ref, artifacts, datasets, environment, model_fields
    )
    privacy_report.entity_audit = _load_entity_audit(root, config)
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
        run=_run_info(run),
        environment=environment,
        metrics=_merge_metrics(run, config),
        privacy_report=privacy_report,
        security_report=security_report,
        declared=config.declared,
        lineage_links=config.build.lineage_links,
    )
    if previous is not None:
        passport.revision = compute_revision(previous, passport, reason)
    if policy is not None:
        passport.policy = evaluate(policy[0], policy[1], passport)
    return sign_passport(passport, private_key)


def _load_entity_audit(root: Path, config: ProjectConfig) -> EntityAuditResult | None:
    path = config.privacy.entity_audit
    if path is None:
        return None
    try:
        return EntityAuditResult.model_validate_json((root / path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BuildError(f"cannot read entity audit {path}: {exc}") from exc


def _collect_datasets(manifest: _Manifest, config: ProjectConfig) -> list[tuple[str, DatasetInfo]]:
    datasets = []
    for ds in config.build.datasets:
        ref = manifest.add(ds.path, ArtifactKind.DATASET)
        info = DatasetInfo(
            name=ds.name or Path(ds.path).stem,
            source=ds.source,
            license=ds.license,
            sha256=ref.sha256,
            split_role=ds.split_role,
        )
        datasets.append((ref.path, info))
    return datasets


def _load_project_policy(
    root: Path, config: ProjectConfig, manifest: _Manifest
) -> tuple[Policy, str] | None:
    """The policy and its hash; the policy file itself becomes a signed artifact."""
    if config.policy is None:
        return None
    ref = manifest.add(config.policy, ArtifactKind.CONFIG)
    try:
        return load_policy(root / ref.path)
    except PolicyError as exc:
        raise BuildError(str(exc)) from exc


def _model_fields(root: Path, config: ProjectConfig, manifest: _Manifest) -> dict[str, Any]:
    """Declared model fields, with values from the train stage's metadata file filling gaps."""
    model = config.build.model
    fields = model.model_dump(exclude={"path", "metadata"}, exclude_defaults=True)
    if model.metadata is None:
        return fields
    manifest.add(model.metadata, ArtifactKind.OTHER)
    metadata = json.loads((root / model.metadata).read_text(encoding="utf-8"))
    return {**metadata, **fields}


def _load_signing_key(root: Path, config: ProjectConfig) -> Ed25519PrivateKey:
    path = root / config.signing.private_key
    if not path.is_file():
        raise BuildError(
            f"signing key not found: {config.signing.private_key} (run `passport init`)"
        )
    return identity.load_private_key(path)


def _run_checks(
    root: Path,
    config: ProjectConfig,
    model_ref: ArtifactRef,
    artifacts: list[ArtifactRef],
    datasets: list[tuple[str, DatasetInfo]],
    environment: Environment,
    model_fields: dict[str, Any],
) -> tuple[PrivacyReport, SecurityReport]:
    security = SecurityReport()
    try:
        privacy = assess_privacy(root, config.privacy, datasets)
        privacy.leakage = assess_models(
            root,
            config.audit,
            model_ref.path,
            artifacts,
            datasets,
            environment,
            security,
            input_schema=model_fields.get("input_schema"),
        )
    except AssessmentError as exc:
        raise BuildError(str(exc)) from exc
    if config.privacy.scan_secrets:
        security.files_scanned_for_secrets, security.secret_findings = scan_secrets(root, artifacts)
    return privacy, security


def _merge_metrics(run: RunRecord | None, config: ProjectConfig) -> dict[str, dict[str, float]]:
    """Metrics captured by the run, overridden by any declared in passport.yaml."""
    metrics = {split: dict(values) for split, values in (run.metrics if run else {}).items()}
    for split, values in config.build.metrics.items():
        metrics.setdefault(split, {}).update(values)
    return metrics


def _run_info(run: RunRecord | None) -> RunInfo | None:
    if run is None:
        return None
    return RunInfo(
        run_id=run.run_id,
        started_at=run.started_at,
        ended_at=run.ended_at,
        overrides=run.overrides,
        mlflow_run_id=run.mlflow_run_id,
    )


def sign_passport(passport: Passport, private_key: Ed25519PrivateKey) -> Passport:
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
