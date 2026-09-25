"""Language model commands: entity-level leakage audits and remediation.

The deep-learning stack is imported inside each command, so the rest of the CLI starts fast
and works without the ``llm`` extra installed.
"""

# ruff: noqa: PLC0415 - optional heavy dependencies are imported where they are used

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from model_passport.cli._app import EXIT_FAIL, LLM, app, fail
from model_passport.core.schema import Severity, at_least

if TYPE_CHECKING:
    from model_passport.core.schema import EntityAuditResult
    from model_passport.llm.entities import Record

llm_app = typer.Typer(help="Audit language models for memorized PII, and remediate.",
                      no_args_is_help=True)  # fmt: skip
app.add_typer(llm_app, name="llm", rich_help_panel=LLM)

CorpusOption = Annotated[
    Path, typer.Option(help="Training corpus: JSONL (text + entities, or AI4Privacy) or .txt.")
]


def _records(corpus: Path) -> list[Record]:
    from model_passport.llm.stages import StageError, read_corpus

    try:
        return read_corpus(corpus)
    except StageError as exc:
        fail(str(exc))


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(v.strip() for v in (value or "").split(",") if v.strip())


def _print_audit(result: EntityAuditResult, top: int = 5) -> None:
    auc = "n/a" if result.auc is None else f"{result.auc:.3f}"
    tpr = result.tpr_at_fpr.get("0.01")
    typer.echo(f"model: {result.model} ({result.access}); primary method: {result.primary_method}")
    typer.echo(
        f"entities audited: {result.entities_audited}; attack AUC {auc}"
        + ("" if tpr is None else f", TPR at 1% FPR {tpr:.3f}")
        + " (chance: AUC 0.5, TPR 0.01)"
    )
    counts = result.severity_counts
    typer.echo(
        "severity: "
        + ", ".join(
            f"{level} {counts.get(level, 0)}" for level in ("critical", "high", "medium", "low")
        )
    )
    methods = [m for m in result.metrics if m.entity_type is None]
    if len(methods) > 1:
        typer.echo("methods: " + ", ".join(f"{m.method} {m.auc:.3f}" for m in methods))
    for f in result.findings[:top]:
        label = f"[{f.severity.value.upper()}]"
        proof = f"  confirmed, exposure {f.exposure:.1f} bits" if f.confirmed and f.exposure else ""
        typer.echo(
            f"  {label:10} {f.entity_type:16} {f.masked_value:24} "
            f"risk {f.risk:4.1f}  record {f.record}{proof}"
        )


@llm_app.command("audit")
def audit(
    model: Annotated[
        str,
        typer.Option(
            help="hf:<id or dir>, a local directory, openai-compatible:<url>#<model>, "
            "anthropic:<model>, or openai:<model>."
        ),
    ],
    corpus: CorpusOption,
    out: Annotated[Path, typer.Option(help="Audit JSON to write.")] = Path(
        "reports/entity_audit.json"
    ),
    references: Annotated[int, typer.Option(help="Alternatives per value (paper: 5).")] = 5,
    source: Annotated[
        str, typer.Option(help="Where alternatives come from: synthetic, corpus, or mix.")
    ] = "synthetic",
    max_entities: Annotated[int, typer.Option(help="Entities to sample and audit.")] = 2000,
    types: Annotated[str | None, typer.Option(help="Only these types, e.g. EMAIL,SSN.")] = None,
    primary: Annotated[
        str, typer.Option(help="Method for per-entity risk, or auto (the strongest).")
    ] = "auto",
    fdr: Annotated[float, typer.Option(help="False discovery rate for flagging.")] = 0.05,
    probe_samples: Annotated[
        int, typer.Option(help="Completions per value for generation-only models.")
    ] = 5,
    fail_on: Annotated[
        str | None, typer.Option(help="Exit 1 if any entity is at least: low ... critical.")
    ] = None,
    device: Annotated[str | None, typer.Option(help="cpu, cuda, or mps (default: auto).")] = None,
) -> None:
    """Entity-level membership inference audit (EL-MIA) of a model on its training data."""
    from model_passport.llm.audit import AuditSettings
    from model_passport.llm.backends import ModelError
    from model_passport.llm.stages import StageError, audit_corpus, fingerprint_key

    settings = AuditSettings(
        references=references, reference_source=source, max_entities=max_entities,
        types=_csv(types), primary=primary, fdr=fdr, probe_samples=probe_samples,
        fingerprint_key=fingerprint_key(),
    )  # fmt: skip
    try:
        result = audit_corpus(model, corpus, out, settings, device)
    except (StageError, ModelError, ValueError) as exc:
        fail(str(exc))
    _print_audit(result)
    typer.echo(f"wrote {out}")
    if fail_on and any(at_least(f.severity, Severity(fail_on)) for f in result.findings):
        raise typer.Exit(EXIT_FAIL)


@llm_app.command("sanitize")
def sanitize(
    corpus: CorpusOption,
    out: Annotated[Path, typer.Option(help="Sanitized corpus (JSONL) to write.")],
    audit_file: Annotated[
        Path | None, typer.Option("--audit", help="Audit JSON whose findings to remove.")
    ] = None,
    min_severity: Annotated[
        str, typer.Option(help="Sanitize findings at least this severe.")
    ] = "medium",
    always: Annotated[
        str | None,
        typer.Option(
            help="Entity types to sanitize regardless of risk, e.g. SSN,CREDITCARDNUMBER."
        ),
    ] = None,
    strategy: Annotated[str, typer.Option(help="surrogate, mask, or drop.")] = "surrogate",
    seed: int = 0,
) -> None:
    """Remove risky entities from a training corpus before retraining."""
    from model_passport.core.schema import EntityAuditResult
    from model_passport.llm.entities import write_corpus
    from model_passport.llm.sanitize import risky_spans
    from model_passport.llm.sanitize import sanitize as run_sanitize

    records = _records(corpus)
    result = None
    if audit_file is not None:
        try:
            result = EntityAuditResult.model_validate_json(audit_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            fail(f"cannot read audit {audit_file}: {exc}")
    targets = risky_spans(result, records, Severity(min_severity), _csv(always))
    try:
        cleaned, report = run_sanitize(records, targets, strategy, seed)
    except ValueError as exc:
        fail(str(exc))
    write_corpus(cleaned, out)
    summary = ", ".join(f"{k} {v}" for k, v in sorted(report.replaced.items())) or "nothing"
    typer.echo(f"{strategy}: replaced {summary}; dropped {report.dropped_records} records")
    typer.echo(f"wrote {out}")


@llm_app.command("finetune")
def finetune(
    corpus: CorpusOption,
    out: Annotated[Path, typer.Option(help="Directory for the fine-tuned model (safetensors).")],
    base: Annotated[
        str, typer.Option(help="Base checkpoint (hf:<id> or a directory), or 'tiny' for demos.")
    ] = "tiny",
    epochs: int = 3,
    learning_rate: Annotated[float, typer.Option("--lr")] = 5e-5,
    batch_size: int = 8,
    seed: int = 0,
    device: Annotated[str | None, typer.Option(help="cpu, cuda, or mps (default: auto).")] = None,
) -> None:
    """Fine-tune a causal language model on a corpus (the retrain step of remediation)."""
    from model_passport.llm.backends import ModelError
    from model_passport.llm.stages import StageError, finetune_corpus
    from model_passport.llm.training import TrainSettings

    settings = TrainSettings(epochs, learning_rate, batch_size, seed=seed, device=device)
    try:
        history = finetune_corpus(corpus, out, base, settings)
    except (StageError, ModelError) as exc:
        fail(str(exc))
    losses = " -> ".join(f"{h:.3f}" for h in history)
    typer.echo(f"trained for {epochs} epochs; loss {losses}")
    typer.echo(f"wrote {out}")


@llm_app.command("demo-corpus")
def demo_corpus(
    out: Annotated[Path, typer.Option(help="JSONL corpus to write.")] = Path("data/corpus.jsonl"),
    records: Annotated[int, typer.Option(help="Number of records.")] = 400,
    seed: int = 0,
) -> None:
    """Write a synthetic corpus (support tickets with fake PII) for demos and tests."""
    from model_passport.llm.entities import write_corpus
    from model_passport.llm.synthetic import corpus

    write_corpus(corpus(records, seed), out)
    typer.echo(f"wrote {records} synthetic records to {out}")


@llm_app.command("remediate")
def remediate(
    config: Annotated[Path, typer.Option(help="Project config file.")] = Path("passport.yaml"),
    max_rounds: Annotated[int, typer.Option(help="Sanitize-and-retrain rounds at most.")] = 3,
    min_severity: Annotated[
        str, typer.Option(help="Sanitize findings at least this severe.")
    ] = "medium",
    always: Annotated[
        str, typer.Option(help="Entity types never to train on, comma separated.")
    ] = "SSN,CREDITCARDNUMBER",
    strategy: Annotated[str, typer.Option(help="surrogate, mask, or drop.")] = "surrogate",
) -> None:
    """Build, and while the release gate fails: sanitize, retrain, re-audit as a new version.

    Every round is a signed passport linked to the one before; old passports and training data
    are archived under .passport/history and data/history.
    """
    from model_passport.core.builder import BuildError
    from model_passport.core.capture import CaptureError
    from model_passport.llm.remediate import (
        RemediationError,
        RemediationSettings,
        gate_passed,
    )
    from model_passport.llm.remediate import remediate as run_remediate

    settings = RemediationSettings(
        max_rounds=max_rounds, min_severity=Severity(min_severity), always=_csv(always),
        strategy=strategy,
    )  # fmt: skip
    try:
        rounds = run_remediate(config, settings, echo=typer.echo)
    except (RemediationError, BuildError, CaptureError, ValueError) as exc:
        fail(str(exc))
    if not gate_passed(rounds):
        raise typer.Exit(EXIT_FAIL)
