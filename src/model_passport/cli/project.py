"""Project lifecycle commands: init, run, build, verify, report, export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from model_passport.cli._app import (
    DEFAULT_CONFIG,
    EXIT_FAIL,
    ConfigOption,
    app,
    fail,
    load_project,
    warn,
)
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
)
from model_passport.core.jsonld import to_jsonld
from model_passport.core.revision import archive, load_previous
from model_passport.core.schema import Passport, PipelineStage, Verdict
from model_passport.core.tracking import MlflowTracker, TrackingUnavailable, dvc_add, log_passport
from model_passport.core.verifier import ArtifactStatus, VerificationReport, verify_passport
from model_passport.report.render import write_html

GITIGNORE_ENTRY = ".passport/"
PassportArgument = Annotated[Path, typer.Argument(help="Passport JSON file.")]
DEFAULT_PASSPORT = Path("passport.json")


def _ensure_gitignored(root: Path) -> bool:
    """Add the key directory to .gitignore. Returns True if the file was changed."""
    gitignore = root / ".gitignore"
    lines = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    if GITIGNORE_ENTRY in (line.strip() for line in lines):
        return False
    lines += ["", "# Model Passport signing keys (never commit)", GITIGNORE_ENTRY]
    gitignore.write_text("\n".join(lines).lstrip("\n") + "\n", encoding="utf-8")
    return True


def _read_passport(path: Path) -> Passport:
    try:
        return Passport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        fail(f"cannot read passport {path}: {exc}")


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


def _start_tracker(cfg: ProjectConfig, root: Path) -> MlflowTracker | None:
    if not cfg.tracking.mlflow_uri:
        return None
    try:
        tracker = MlflowTracker(cfg.tracking, root, cfg.project.name)
        tracker.start(git_commit(root))
    except TrackingUnavailable as exc:
        warn(f"MLflow logging disabled: {exc}")
        return None
    return tracker


@app.command()
def run(
    config: ConfigOption = DEFAULT_CONFIG,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override a stage parameter: stage.key=value (repeatable)."),
    ] = None,
) -> None:
    """Run the pipeline stages from passport.yaml and record their provenance."""
    root = config.parent.resolve()
    cfg = load_project(config)
    tracker = _start_tracker(cfg, root)

    def on_stage(stage: StageConfig, record: PipelineStage, metrics: dict[str, Any]) -> None:
        typer.echo(f"stage {stage.name}: ok ({len(record.outputs)} outputs)")
        if tracker is not None:
            tracker.log_stage(stage, record, metrics)
        if cfg.tracking.dvc:
            try:
                dvc_add(root, stage.outs)
            except TrackingUnavailable as exc:
                warn(f"DVC tracking skipped: {exc}")

    try:
        record = run_pipeline(root, cfg, parse_overrides(set_ or []), on_stage=on_stage)
    except CaptureError as exc:
        if tracker is not None:
            tracker.finish(status="FAILED")
        fail(str(exc))

    if tracker is not None:
        tracker.finish()
        record.mlflow_run_id = tracker.parent_run_id
    path = save_run_record(root, record)
    typer.echo(f"recorded {len(record.stages)} stages in {path.relative_to(root)}")


def _log_to_mlflow(cfg: ProjectConfig, config: Path, passport: Passport, out: Path) -> None:
    if not (passport.run and passport.run.mlflow_run_id):
        return
    tags = {"passport.id": str(passport.identity.passport_id)}
    if passport.policy is not None:
        tags["passport.verdict"] = passport.policy.verdict.value
    try:
        log_passport(cfg.tracking, config.parent.resolve(), passport.run.mlflow_run_id, out, tags)
        typer.echo(f"  mlflow run:  {passport.run.mlflow_run_id}")
    except TrackingUnavailable as exc:
        warn(f"passport not logged to MLflow: {exc}")


def _print_build_summary(passport: Passport, out: Path) -> None:
    typer.echo(f"wrote {out}")
    typer.echo(f"  passport_id: {passport.identity.passport_id}")
    typer.echo(f"  artifacts:   {len(passport.artifacts)}")
    typer.echo(f"  stages:      {len(passport.pipeline)}")
    typer.echo(f"  merkle_root: {passport.identity.merkle_root}")
    revision = passport.revision
    if revision is not None:
        changed = [c.name for c in revision.dataset_changes if c.status.value != "unchanged"]
        typer.echo(
            f"  supersedes:  {revision.previous_passport_id} (revision {revision.sequence}; "
            f"datasets changed: {', '.join(changed) or 'none'})"
        )


def _print_policy(passport: Passport) -> None:
    if passport.policy is None:
        return
    typer.echo("")
    for rule in passport.policy.rules:
        typer.echo(f"  [{rule.result.value.upper():4}] {rule.name}: {rule.message}")
    typer.echo(f"verdict: {passport.policy.verdict.value.upper()}")


@app.command()
def build(
    config: ConfigOption = DEFAULT_CONFIG,
    out: Annotated[Path, typer.Option(help="Output passport path.")] = DEFAULT_PASSPORT,
    previous: Annotated[
        Path | None,
        typer.Option(help="Passport this build supersedes (default: the existing --out file)."),
    ] = None,
    link: Annotated[
        bool, typer.Option(help="Link to and archive the previous passport of this model.")
    ] = True,
    reason: Annotated[str | None, typer.Option(help="Why the model was rebuilt.")] = None,
    html: Annotated[bool, typer.Option(help="Also write an HTML report next to --out.")] = True,
) -> None:
    """Run all checks, apply the policy, sign, and write passport.json (exit 1 on fail)."""
    cfg = load_project(config)
    prior = load_previous(previous or out, cfg.project.name) if link else None
    try:
        passport = build_passport(config, cfg, previous=prior, reason=reason)
    except (OSError, BuildError, ValidationError, ValueError) as exc:
        fail(str(exc))

    if prior is not None and (previous is None or previous.resolve() == out.resolve()):
        archive(config.parent.resolve(), out)
    write_passport(passport, out)
    _print_build_summary(passport, out)
    if html:
        write_html(passport, out.with_suffix(".html"))
        typer.echo(f"  report:      {out.with_suffix('.html')}")
    _log_to_mlflow(cfg, config, passport, out)
    _print_policy(passport)
    if passport.policy is not None and passport.policy.verdict is Verdict.FAIL:
        raise typer.Exit(EXIT_FAIL)


def _print_verification(report: VerificationReport) -> None:
    for check in report.changed:
        detail = "file is missing" if check.status is ArtifactStatus.MISSING else "hash mismatch"
        typer.echo(f"CHANGED  {check.path} ({check.kind.value}): {detail}")
    for problem in report.event_errors:
        typer.echo(f"EVENT    {problem}")
    unchanged = len(report.artifacts) - len(report.changed)
    typer.echo(f"artifacts:   {unchanged}/{len(report.artifacts)} unchanged")
    typer.echo(f"merkle root: {'ok' if report.merkle_ok else 'MISMATCH'}")
    typer.echo(f"key:         {'ok' if report.fingerprint_ok else 'FINGERPRINT MISMATCH'}")
    typer.echo(f"signature:   {'ok' if report.signature_ok else 'INVALID'}")
    typer.echo(f"events:      {'ok' if not report.event_errors else 'CHAIN BROKEN'}")


@app.command()
def verify(
    passport: PassportArgument = DEFAULT_PASSPORT,
    public_key: Annotated[
        Path | None,
        typer.Option(help="Trusted public key (defaults to the project key under --root)."),
    ] = None,
    root: Annotated[
        Path, typer.Option(help="Project root that artifact paths are relative to.")
    ] = Path("."),
) -> None:
    """Recompute hashes, rebuild the Merkle root, and check the signature and events."""
    report = verify_passport(passport, public_key or root / SigningConfig().public_key, root)
    for error in report.errors:
        typer.echo(f"error: {error}", err=True)
    if not report.errors:
        _print_verification(report)
    if not report.ok:
        typer.echo("VERIFICATION FAILED", err=True)
        raise typer.Exit(EXIT_FAIL)
    typer.echo("VERIFIED")


@app.command()
def report(
    passport: PassportArgument = DEFAULT_PASSPORT,
    out: Annotated[Path | None, typer.Option(help="HTML path (default: next to passport).")] = None,
    root: Annotated[
        Path | None, typer.Option(help="Also verify artifacts under this root and show it.")
    ] = None,
    public_key: Annotated[Path | None, typer.Option(help="Key for --root verification.")] = None,
) -> None:
    """Render the passport as a self-contained HTML report."""
    parsed = _read_passport(passport)
    verification = None
    if root is not None:
        key = public_key or root / SigningConfig().public_key
        verification = verify_passport(passport, key, root)
    target = out or passport.with_suffix(".html")
    write_html(parsed, target, verification)
    typer.echo(f"wrote {target}")


@app.command()
def export(
    passport: PassportArgument = DEFAULT_PASSPORT,
    out: Annotated[
        Path | None, typer.Option(help="Output path (default: passport.jsonld).")
    ] = None,
) -> None:
    """Export the passport as JSON-LD with W3C PROV, DCAT, and ML Schema terms."""
    target = out or passport.with_suffix(".jsonld")
    document = to_jsonld(_read_passport(passport))
    target.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    typer.echo(f"wrote {target}")
