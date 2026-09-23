"""Operations commands: monitor (drift), serve (registry), push (upload)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from model_passport.cli._app import EXIT_FAIL, app, fail, warn
from model_passport.core import identity
from model_passport.core.config import SigningConfig
from model_passport.monitoring.drift import DEFAULT_ALPHA, DEFAULT_PSI_THRESHOLD
from model_passport.monitoring.monitor import (
    DEFAULT_DEGRADATION_TOLERANCE,
    MonitorError,
    MonitorResult,
    monitor_batch,
)
from model_passport.registry.client import RegistryClient, RegistryError

monitor_app = typer.Typer(help="Monitor deployed models.", no_args_is_help=True)
app.add_typer(monitor_app, name="monitor")

RegistryOption = Annotated[
    str | None, typer.Option(help="Registry URL (default: $PASSPORT_REGISTRY_URL).")
]
TokenOption = Annotated[
    str | None, typer.Option(help="Registry token (default: $PASSPORT_REGISTRY_TOKEN).")
]


def _print_monitor(result: MonitorResult) -> None:
    drift = result.drift
    typer.echo(f"batch: {drift.rows} rows vs reference {drift.reference_rows} rows")
    schema = drift.schema
    for label, items in (
        ("missing columns", schema.missing_columns),
        ("type mismatches", schema.type_mismatches),
        ("unexpected columns", schema.unexpected_columns),
    ):
        if items:
            typer.echo(f"  schema: {label}: {', '.join(items)}")
    for column, increase in schema.null_rate_increases.items():
        typer.echo(f"  schema: null rate of {column} up {increase:.0%}")
    for feature in drift.features:
        flag = "DRIFT" if feature.drifted else "ok   "
        typer.echo(
            f"  [{flag}] {feature.feature:18} {feature.kind:11} psi={feature.psi:.3f} "
            f"tests rejected={feature.rejected}/{len(feature.tests)}"
        )
    perf = result.performance
    if perf and "accuracy" in perf:
        baseline = "?" if perf["baseline_accuracy"] is None else perf["baseline_accuracy"]
        degraded = " (DEGRADED)" if perf["degraded"] else ""
        typer.echo(f"  live accuracy {perf['accuracy']:.3f} vs test {baseline}{degraded}")
    typer.echo("retraining recommended" if result.retrain_recommended else "no retraining needed")


@monitor_app.command("drift")
def monitor_drift(
    batch: Annotated[Path, typer.Argument(help="Live batch (CSV, Parquet, JSONL).")],
    passport: Annotated[Path, typer.Option(help="Passport to check against.")] = Path(
        "passport.json"
    ),
    root: Annotated[Path, typer.Option(help="Project root holding the reference data.")] = Path(
        "."
    ),
    reference: Annotated[
        str | None, typer.Option(help="Reference dataset name (default: train split).")
    ] = None,
    alpha: Annotated[float, typer.Option(help="Family-wise significance level.")] = DEFAULT_ALPHA,
    psi_threshold: Annotated[
        float, typer.Option(help="Minimum PSI for a significant shift to count as drift.")
    ] = DEFAULT_PSI_THRESHOLD,
    tolerance: Annotated[
        float, typer.Option(help="Allowed accuracy drop on labeled batches.")
    ] = DEFAULT_DEGRADATION_TOLERANCE,
    private_key: Annotated[Path | None, typer.Option(help="Signing key for the event.")] = None,
    registry: RegistryOption = None,
    token: TokenOption = None,
) -> None:
    """Check a batch for schema problems, drift, and degradation; append a signed event.

    Exits 1 when retraining is recommended, so schedulers and CI can trigger `passport run`.
    """
    key_path = private_key or root / SigningConfig().private_key
    try:
        key = identity.load_private_key(key_path)
        result = monitor_batch(
            passport, batch, root, key, reference, alpha, psi_threshold, tolerance
        )
    except (OSError, ValueError, MonitorError) as exc:
        fail(str(exc))
    _print_monitor(result)
    typer.echo(f"appended signed event {result.event['event_id']} to {passport}")

    if registry:
        passport_id = json.loads(passport.read_text(encoding="utf-8"))["identity"]["passport_id"]
        try:
            RegistryClient(registry, token).append_event(passport_id, result.event)
            typer.echo(f"sent event to {registry}")
        except RegistryError as exc:
            warn(str(exc))
    if result.retrain_recommended:
        raise typer.Exit(EXIT_FAIL)


@app.command()
def push(
    passport: Annotated[Path, typer.Argument(help="Passport JSON file.")] = Path("passport.json"),
    public_key: Annotated[Path, typer.Option(help="Public key that signed it.")] = Path(
        ".passport/signing_key.pub"
    ),
    registry: RegistryOption = None,
    token: TokenOption = None,
) -> None:
    """Upload a passport to the registry (the registry re-verifies it before storing)."""
    try:
        document = json.loads(passport.read_text(encoding="utf-8"))
        pem = public_key.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        fail(str(exc))
    client = RegistryClient(registry, token)
    try:
        summary = client.upload(document, pem)
    except RegistryError as exc:
        fail(str(exc), EXIT_FAIL)
    typer.echo(
        f"uploaded {summary['model_name']} {summary['version']} "
        f"({summary['passport_id']}, verdict {summary['verdict']}) to {client.url}"
    )


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8000,
    db: Annotated[Path, typer.Option(help="SQLite database.")] = Path("registry.db"),
    token: TokenOption = None,
    trusted_keys: Annotated[
        Path | None, typer.Option(help="Directory of trusted *.pub keys.")
    ] = None,
) -> None:
    """Run the passport registry API."""
    try:
        import uvicorn

        from model_passport.registry.api import create_app
    except ImportError:
        fail("the registry needs extras: pip install 'model-passport[registry]'")
    application = create_app(db, token, str(trusted_keys) if trusted_keys else None)
    uvicorn.run(application, host=host, port=port)
