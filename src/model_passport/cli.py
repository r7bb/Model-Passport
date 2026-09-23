"""``passport`` command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from model_passport.core import identity
from model_passport.core.builder import BuildError, build_passport, write_passport
from model_passport.core.capture import (
    CaptureError,
    git_commit,
    parse_overrides,
    run_pipeline,
    save_run_record,
)
from model_passport.core.config import (
    CONFIG_FILENAME,
    CONFIG_TEMPLATE,
    ProjectConfig,
    SigningConfig,
    StageConfig,
    load_config,
)
from model_passport.core.schema import PipelineStage
from model_passport.core.tracking import MlflowTracker, TrackingUnavailable, dvc_add, log_passport
from model_passport.core.verifier import ArtifactStatus, verify_passport

app = typer.Typer(help="Signed, verifiable passports for trained ML models.", no_args_is_help=True)

GITIGNORE_ENTRY = ".passport/"


def _ensure_gitignored(root: Path) -> bool:
    """Add the key directory to .gitignore. Returns True if the file was changed."""
    gitignore = root / ".gitignore"
    lines = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    if GITIGNORE_ENTRY in (line.strip() for line in lines):
        return False
    lines += ["", "# Model Passport signing keys (never commit)", GITIGNORE_ENTRY]
    gitignore.write_text("\n".join(lines).lstrip("\n") + "\n", encoding="utf-8")
    return True


@app.command()
def init(
    directory: Annotated[Path, typer.Argument(help="Project directory.")] = Path("."),
    name: Annotated[
        str | None, typer.Option(help="Model name (defaults to directory name).")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing config and signing key.")
    ] = False,
) -> None:
    """Set up a project: passport.yaml, an Ed25519 signing keypair, and .gitignore entries."""
    root = directory.resolve()
    root.mkdir(parents=True, exist_ok=True)

    config_path = root / CONFIG_FILENAME
    if config_path.exists() and not force:
        typer.echo(f"kept existing {CONFIG_FILENAME}")
    else:
        config_path.write_text(CONFIG_TEMPLATE.format(name=name or root.name), encoding="utf-8")
        typer.echo(f"wrote {CONFIG_FILENAME}")

    signing = SigningConfig()
    private_path, public_path = root / signing.private_key, root / signing.public_key
    if private_path.exists() and not force:
        typer.echo(f"kept existing signing key {signing.private_key}")
    else:
        identity.save_keypair(identity.generate_keypair(), private_path, public_path)
        typer.echo(f"generated signing key {signing.private_key}")

    fingerprint = identity.public_key_fingerprint(identity.load_public_key(public_path))
    typer.echo(f"public key fingerprint: {fingerprint}")
    if _ensure_gitignored(root):
        typer.echo(f"added {GITIGNORE_ENTRY} to .gitignore")


def _warn(message: str) -> None:
    typer.echo(f"warning: {message}", err=True)


def _load(config: Path) -> ProjectConfig:
    try:
        return load_config(config)
    except (OSError, ValidationError, yaml.YAMLError) as exc:
        typer.echo(f"error: cannot load {config}: {exc}", err=True)
        raise typer.Exit(2) from exc


@app.command()
def run(
    config: Annotated[Path, typer.Option(help="Project config file.")] = Path(CONFIG_FILENAME),
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override a stage parameter: stage.key=value (repeatable)."),
    ] = None,
) -> None:
    """Run the pipeline stages from passport.yaml and record their provenance."""
    root = config.parent.resolve()
    cfg = _load(config)

    tracker: MlflowTracker | None = None
    if cfg.tracking.mlflow_uri:
        try:
            tracker = MlflowTracker(cfg.tracking, root, cfg.project.name)
            tracker.start(git_commit(root))
        except TrackingUnavailable as exc:
            _warn(f"MLflow logging disabled: {exc}")
            tracker = None

    def on_stage(stage: StageConfig, record: PipelineStage, metrics: dict) -> None:
        typer.echo(f"stage {stage.name}: ok ({len(record.outputs)} outputs)")
        if tracker is not None:
            tracker.log_stage(stage, record, metrics)
        if cfg.tracking.dvc:
            try:
                dvc_add(root, stage.outs)
            except TrackingUnavailable as exc:
                _warn(f"DVC tracking skipped: {exc}")

    try:
        record = run_pipeline(root, cfg, parse_overrides(set_ or []), on_stage=on_stage)
    except CaptureError as exc:
        if tracker is not None:
            tracker.finish(status="FAILED")
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    if tracker is not None:
        tracker.finish()
        record.mlflow_run_id = tracker.parent_run_id
    path = save_run_record(root, record)
    typer.echo(f"recorded {len(record.stages)} stages in {path.relative_to(root)}")


@app.command()
def build(
    config: Annotated[Path, typer.Option(help="Project config file.")] = Path(CONFIG_FILENAME),
    out: Annotated[Path, typer.Option(help="Output passport path.")] = Path("passport.json"),
) -> None:
    """Hash all artifacts, compute the Merkle root, sign, and write passport.json."""
    cfg = _load(config)
    try:
        passport = build_passport(config, cfg)
    except (OSError, BuildError, ValidationError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    write_passport(passport, out)
    typer.echo(f"wrote {out}")
    typer.echo(f"  passport_id: {passport.identity.passport_id}")
    typer.echo(f"  artifacts:   {len(passport.artifacts)}")
    typer.echo(f"  stages:      {len(passport.pipeline)}")
    typer.echo(f"  merkle_root: {passport.identity.merkle_root}")

    if passport.run and passport.run.mlflow_run_id:
        try:
            log_passport(
                cfg.tracking,
                config.parent.resolve(),
                passport.run.mlflow_run_id,
                out,
                {"passport.id": str(passport.identity.passport_id)},
            )
            typer.echo(f"  mlflow run:  {passport.run.mlflow_run_id}")
        except TrackingUnavailable as exc:
            _warn(f"passport not logged to MLflow: {exc}")


@app.command()
def verify(
    passport: Annotated[Path, typer.Argument(help="Passport JSON file.")] = Path("passport.json"),
    public_key: Annotated[
        Path | None,
        typer.Option(help="Trusted public key (defaults to the project key under --root)."),
    ] = None,
    root: Annotated[Path, typer.Option(help="Project root artifact paths are relative to.")] = Path(
        "."
    ),
) -> None:
    """Recompute hashes, rebuild the Merkle root, and check the signature."""
    key_path = public_key or root / SigningConfig().public_key
    report = verify_passport(passport, key_path, root)

    for error in report.errors:
        typer.echo(f"error: {error}", err=True)
    for check in report.artifacts:
        if check.status is ArtifactStatus.OK:
            continue
        detail = "file is missing" if check.status is ArtifactStatus.MISSING else "hash mismatch"
        typer.echo(f"CHANGED  {check.path} ({check.kind.value}): {detail}")
    if not report.errors:
        typer.echo(
            f"artifacts:   {len(report.artifacts) - len(report.changed)}/"
            f"{len(report.artifacts)} unchanged"
        )
        typer.echo(f"merkle root: {'ok' if report.merkle_ok else 'MISMATCH'}")
        typer.echo(f"key:         {'ok' if report.fingerprint_ok else 'FINGERPRINT MISMATCH'}")
        typer.echo(f"signature:   {'ok' if report.signature_ok else 'INVALID'}")

    if report.ok:
        typer.echo("VERIFIED")
    else:
        typer.echo("VERIFICATION FAILED", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
