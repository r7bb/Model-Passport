"""``passport scan`` and ``passport audit``: run scanners and auditors on individual files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from model_passport.auditors.artifact import (
    UnsafeArtifactError,
    audit_dependencies,
    safe_load_pickle,
    scan_model_file,
)
from model_passport.auditors.leakage import LeakageAuditError, loss_threshold_attack
from model_passport.cli._app import CHECKS, EXIT_FAIL, app, fail
from model_passport.core.capture import capture_environment
from model_passport.core.schema import DependencyAudit, Finding, Severity, at_least
from model_passport.scanners.base import ScanError, ScanTarget, load_table
from model_passport.scanners.data_pii import PiiScanner
from model_passport.scanners.reid_risk import ReidRiskScanner
from model_passport.scanners.secrets import SecretsScanner

scan_app = typer.Typer(
    help="Scan data and code for privacy and secret leaks.", no_args_is_help=True
)
audit_app = typer.Typer(help="Audit models and dependencies.", no_args_is_help=True)
app.add_typer(scan_app, name="scan", rich_help_panel=CHECKS)
app.add_typer(audit_app, name="audit", rich_help_panel=CHECKS)

SEVERITY_CHOICES = [s.value for s in Severity]
OutOption = Annotated[Path | None, typer.Option(help="Write findings as JSON.")]
FailOnOption = Annotated[
    str | None,
    typer.Option(help=f"Exit 1 if any finding is at least this severity: {SEVERITY_CHOICES}."),
]


def _print_findings(findings: list[Finding]) -> None:
    if not findings:
        typer.echo("no findings")
    for f in findings:
        examples = f"  e.g. {', '.join(f.masked_examples)}" if f.masked_examples else ""
        typer.echo(f"  [{f.severity.value:8}] {f.category:22} {f.location}  n={f.count}{examples}")


def _finish(findings: list[Finding], out: Path | None, fail_on: str | None) -> None:
    _print_findings(findings)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = [f.model_dump(mode="json") for f in findings]
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}")
    if fail_on and any(at_least(f.severity, Severity(fail_on)) for f in findings):
        raise typer.Exit(EXIT_FAIL)


@scan_app.command("data")
def scan_data(
    path: Annotated[Path, typer.Argument(help="CSV, TSV, Parquet, or JSONL file.")],
    quasi: Annotated[
        str | None, typer.Option(help="Comma-separated quasi-identifiers (default: suggested).")
    ] = None,
    sensitive: Annotated[str | None, typer.Option(help="Sensitive column for l-diversity.")] = None,
    sample_size: Annotated[int, typer.Option(help="Rows sampled for PII scanning.")] = 10_000,
    full: Annotated[bool, typer.Option("--full", help="Scan every row for PII.")] = False,
    presidio: Annotated[bool, typer.Option(help="Also use Presidio on free text.")] = False,
    out: OutOption = None,
    fail_on: FailOnOption = None,
) -> None:
    """PII and reidentification risk scan of a table."""
    target = ScanTarget(path)
    try:
        frame = target.frame
    except ScanError as exc:
        fail(str(exc))
    typer.echo(f"{path}: {len(frame)} rows, {len(frame.columns)} columns")
    pii = PiiScanner(sample_size=None if full else sample_size, use_presidio=presidio)
    qis = [q.strip() for q in quasi.split(",")] if quasi else None
    reid = ReidRiskScanner(qis, sensitive)
    _finish(pii.scan(target) + reid.scan(target), out, fail_on)


@scan_app.command("secrets")
def scan_secrets(
    paths: Annotated[list[Path], typer.Argument(help="Files or directories to scan.")],
    out: OutOption = None,
    fail_on: FailOnOption = "high",
) -> None:
    """Scan files for API keys, tokens, and private keys."""
    scanner = SecretsScanner()
    files = [
        file
        for path in paths
        for file in ([path] if path.is_file() else sorted(path.rglob("*")))
        if file.is_file() and ".git" not in file.parts
    ]
    findings = [f for file in files for f in scanner.scan(ScanTarget(file, name=str(file)))]
    typer.echo(f"scanned {len(files)} files")
    _finish(findings, out, fail_on)


@audit_app.command("model")
def audit_model(
    model: Annotated[Path, typer.Argument(help="Serialized scikit-learn model.")],
    members: Annotated[Path, typer.Option(help="Training data (members).")],
    nonmembers: Annotated[Path, typer.Option(help="Held-out data (non-members).")],
    label: Annotated[str, typer.Option(help="Label column.")],
    string_cols: Annotated[
        str | None, typer.Option(help="Comma-separated columns to read as text.")
    ] = None,
    out: OutOption = None,
) -> None:
    """Pickle safety scan plus loss-threshold membership inference and generalization gap."""
    findings = scan_model_file(model)
    _print_findings(findings)
    try:
        estimator = safe_load_pickle(model)
    except UnsafeArtifactError as exc:
        fail(str(exc), EXIT_FAIL)
    text = [c.strip() for c in string_cols.split(",")] if string_cols else None
    try:
        result = loss_threshold_attack(
            estimator, load_table(members, text), load_table(nonmembers, text), label
        )
    except (LeakageAuditError, ScanError) as exc:
        fail(str(exc))
    typer.echo(f"membership inference AUC: {result.mia_auc:.4f} (0.5 = no leakage)")
    typer.echo(f"TPR at {result.low_fpr:.0%} FPR:        {result.tpr_at_low_fpr:.4f}")
    typer.echo(f"generalization gap:       {result.generalization_gap:+.4f} ({result.gap_metric})")
    if out is not None:
        document = {
            "leakage": result.model_dump(mode="json"),
            "findings": [f.model_dump(mode="json") for f in findings],
        }
        out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}")


@audit_app.command("deps")
def audit_deps(
    out: OutOption = None,
    fail_on: FailOnOption = "critical",
) -> None:
    """Run pip-audit on the current environment, with severities from OSV."""
    status, message, findings = audit_dependencies(capture_environment().dependencies)
    typer.echo(message)
    if status is not DependencyAudit.OK:
        fail(message)
    _finish(findings, out, fail_on)
