"""Workers: the automated steps of the lifecycle, run as background jobs (M7).

- ``train``: fine-tune the model's base checkpoint on the version's training data.
- ``audit``: run the entity-level audit against the reference data, apply the release gate,
  sign an attestation, and move the version to ``findings`` or ``clean``.
- ``remediate``: sanitize the training data (keeping provenance), retrain, register the next
  version as the parent's successor, and queue its audit.
- ``report``: generate the diligence report and store it encrypted.

A job is claimed in a short transaction; the heavy work runs outside any transaction, and the
result is written in a second one. Datasets and models are decrypted into a private temporary
directory that is deleted when the job ends.
"""

from __future__ import annotations

import io
import json
import socket
import tarfile
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from model_passport.core import identity
from model_passport.core.schema import EntityAuditResult, Severity
from model_passport.llm.audit import AuditSettings
from model_passport.llm.entities import load_corpus, write_corpus
from model_passport.llm.remediate import bump_minor, translate
from model_passport.llm.sanitize import risky_spans, sanitize
from model_passport.llm.stages import audit_corpus, finetune_corpus
from model_passport.llm.training import TrainSettings
from model_passport.platform import attest, gate, jobs, lifecycle, reports, services
from model_passport.platform.auditlog import SYSTEM
from model_passport.platform.db import ALL_TENANTS, scoped_session
from model_passport.platform.models import (
    Access,
    Dataset,
    Job,
    Model,
    ModelVersion,
    State,
    Tenant,
)
from model_passport.platform.storage import ObjectStore, TenantStore

WORKER = lifecycle.Who(SYSTEM, None, system=True)
ALWAYS_SANITIZE = ("SSN", "CREDITCARDNUMBER")


class JobFailedError(RuntimeError):
    """The job cannot complete; the message is recorded on the job."""


@dataclass
class WorkerContext:
    factory: sessionmaker[Session]
    store: ObjectStore
    master_key: bytes
    name: str = field(default_factory=lambda: f"{socket.gethostname()}-worker")
    audit: AuditSettings = field(default_factory=AuditSettings)
    training: dict[str, TrainSettings] = field(
        default_factory=lambda: {
            "tiny": TrainSettings(epochs=15, learning_rate=2e-3, batch_size=32),
            "checkpoint": TrainSettings(epochs=3, learning_rate=5e-5, batch_size=8),
        }
    )

    def files(self, tenant: Tenant) -> TenantStore:
        return services.tenant_store(self.store, self.master_key, tenant)


# --- Helpers ---------------------------------------------------------------------------------


def _pack(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(directory.rglob("*")):
            archive.add(path, arcname=path.relative_to(directory).as_posix(), recursive=False)
    return buffer.getvalue()


def _unpack(data: bytes, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        archive.extractall(target, filter="data")  # rejects absolute paths and traversal
    return target


def _model_spec(files: TenantStore, version: ModelVersion, work: Path) -> str:
    """A loadable model: the version's own checkpoint if it has one, else the base spec."""
    if version.artifact_key:
        return str(_unpack(files.get(version.artifact_key), work / "model"))
    base = version.model.base
    if base == "tiny":
        raise JobFailedError("a tiny demo model must be trained before it can be audited")
    return base


def _training(ctx: WorkerContext, base: str) -> TrainSettings:
    return ctx.training["tiny" if base == "tiny" else "checkpoint"]


def _store_model(files: TenantStore, directory: Path) -> tuple[str, str]:
    digest = identity.sha256_path(directory)
    key = f"models/{digest}.tar.gz"
    files.put(key, _pack(directory))
    return key, digest


# --- Handlers --------------------------------------------------------------------------------


def _train(ctx: WorkerContext, job: Job, work: Path) -> dict[str, Any]:
    with scoped_session(ctx.factory, job.tenant_id) as session:
        version = services.get(session, ModelVersion, job.tenant_id, job.payload["version"])
        tenant = session.get(Tenant, job.tenant_id)
        assert tenant is not None
        if version.model.access is not Access.OPEN_WEIGHT:
            raise JobFailedError("only open-weight models can be trained on the platform")
        dataset = session.get(Dataset, version.dataset_id) if version.dataset_id else None
        if dataset is None:
            raise JobFailedError("the version has no training data")
        files = ctx.files(tenant)
        corpus = work / "train.jsonl"
        corpus.write_bytes(files.get(dataset.object_key))
        base = version.model.base
    out = work / "trained"
    history = finetune_corpus(corpus, out, base, _training(ctx, base))
    with scoped_session(ctx.factory, job.tenant_id) as session:
        version = services.get(session, ModelVersion, job.tenant_id, job.payload["version"])
        version.artifact_key, version.artifact_sha256 = _store_model(files, out)
        if job.payload.get("then_audit"):
            lifecycle.move(session, version, State.AUDITING, WORKER, "trained; auditing")
            jobs.enqueue(session, job.tenant_id, "audit", {"version": version.id}, SYSTEM)
    return {"loss": [round(h, 4) for h in history], "artifact_sha256": version.artifact_sha256}


def _audit(ctx: WorkerContext, job: Job, work: Path) -> dict[str, Any]:
    with scoped_session(ctx.factory, job.tenant_id) as session:
        version = services.get(session, ModelVersion, job.tenant_id, job.payload["version"])
        tenant = session.get(Tenant, job.tenant_id)
        assert tenant is not None
        reference = session.get(Dataset, version.reference_dataset_id or "")
        if reference is None:
            raise JobFailedError("the version has no reference dataset to audit against")
        files = ctx.files(tenant)
        corpus = work / "reference.jsonl"
        corpus.write_bytes(files.get(reference.object_key))
        spec = _model_spec(files, version, work)
    result = audit_corpus(spec, corpus, work / "audit.json", ctx.audit)
    verdict = gate.check(result)
    with scoped_session(ctx.factory, job.tenant_id) as session:
        version = services.get(session, ModelVersion, job.tenant_id, job.payload["version"])
        tenant = session.get(Tenant, job.tenant_id)
        assert tenant is not None
        version.audit = result.model_dump(mode="json")
        version.auc = result.auc
        version.critical = result.severity_counts.get("critical", 0)
        version.high = result.severity_counts.get("high", 0)
        version.verdict = verdict.verdict.value
        risky = gate.blocking(verdict) or version.critical + version.high > 0
        target = State.FINDINGS if risky else State.CLEAN
        lifecycle.move(session, version, target, WORKER, f"gate verdict {version.verdict}")
        attest.attest(session, tenant, ctx.files(tenant), version, verdict)
    return {"verdict": verdict.verdict.value, "auc": result.auc,
            "severity_counts": result.severity_counts}  # fmt: skip


def _whole_types(session: Session, version: ModelVersion, audit: EntityAuditResult) -> set[str]:
    """Types flagged again after a previous remediation round are replaced entirely."""
    risky = {
        f.entity_type for f in audit.findings if f.severity in (Severity.HIGH, Severity.CRITICAL)
    }
    parent = session.get(ModelVersion, version.parent_id) if version.parent_id else None
    if parent is None or not parent.audit:
        return set()
    before = EntityAuditResult.model_validate(parent.audit)
    return risky & {
        f.entity_type for f in before.findings if f.severity in (Severity.HIGH, Severity.CRITICAL)
    }


def _remediate(ctx: WorkerContext, job: Job, work: Path) -> dict[str, Any]:
    with scoped_session(ctx.factory, job.tenant_id) as session:
        version = services.get(session, ModelVersion, job.tenant_id, job.payload["version"])
        tenant = session.get(Tenant, job.tenant_id)
        assert tenant is not None
        if version.model.access is not Access.OPEN_WEIGHT:
            raise JobFailedError(
                "API models cannot be retrained here; sanitize the data and resubmit"
            )
        if not version.audit:
            raise JobFailedError("audit the version before remediating it")
        training = session.get(Dataset, version.dataset_id or "")
        reference = session.get(Dataset, version.reference_dataset_id or "")
        if training is None or reference is None:
            raise JobFailedError("the version needs training and reference datasets")
        files = ctx.files(tenant)
        audit = EntityAuditResult.model_validate(version.audit)
        whole = _whole_types(session, version, audit)
        (work / "train.jsonl").write_bytes(files.get(training.object_key))
        (work / "raw.jsonl").write_bytes(files.get(reference.object_key))
        base, next_version = version.model.base, bump_minor(version.version)
        source = {"name": training.name, "license": training.license, "consent": training.consent}
    raw = list(load_corpus(work / "raw.jsonl"))
    train = list(load_corpus(work / "train.jsonl"))
    found = risky_spans(audit, raw, Severity.MEDIUM)
    for record_id, spans in risky_spans(
        None, raw, Severity.MEDIUM, (*ALWAYS_SANITIZE, *whole)
    ).items():
        found.setdefault(record_id, set()).update(spans)
    cleaned, report = sanitize(train, translate(found, raw, train, only_original=True))
    write_corpus(cleaned, work / "sanitized.jsonl")
    out = work / "retrained"
    finetune_corpus(work / "sanitized.jsonl", out, base, _training(ctx, base))
    with scoped_session(ctx.factory, job.tenant_id) as session:
        version = services.get(session, ModelVersion, job.tenant_id, job.payload["version"])
        tenant = session.get(Tenant, job.tenant_id)
        assert tenant is not None
        files = ctx.files(tenant)
        dataset = services.add_dataset(
            session, files, tenant, f"{source['name']}-v{next_version}",
            (work / "sanitized.jsonl").read_bytes(),
            services.Provenance(f"sanitized from {source['name']} for v{next_version}",
                                source["license"], source["consent"]),
            SYSTEM,
        )  # fmt: skip
        reference_row = session.get(Dataset, version.reference_dataset_id or "")
        assert reference_row is not None
        successor = services.register_version(
            session, version.model, next_version, dataset, reference_row, SYSTEM,
            artifact=_store_model(files, out), parent=version,
        )  # fmt: skip
        lifecycle.move(session, version, State.SUPERSEDED, WORKER, f"replaced by v{next_version}")
        lifecycle.move(session, successor, State.AUDITING, WORKER, "remediated; auditing")
        jobs.enqueue(session, job.tenant_id, "audit", {"version": successor.id}, SYSTEM)
    return {"successor": next_version, "sanitized": dict(report.replaced),
            "whole_types": sorted(whole)}  # fmt: skip


def _report(ctx: WorkerContext, job: Job, _work: Path) -> dict[str, Any]:
    with scoped_session(ctx.factory, job.tenant_id) as session:
        model = services.get(session, Model, job.tenant_id, job.payload["model"])
        tenant = session.get(Tenant, job.tenant_id)
        assert tenant is not None
        report = reports.diligence(session, tenant, model)
        files = ctx.files(tenant)
        stamp = report["generated_at"].replace(":", "")
        files.put(f"reports/{model.id}/{stamp}.json", json.dumps(report, indent=2).encode())
        files.put(f"reports/{model.id}/{stamp}.html", reports.render_html(report).encode())
    return {"json": f"reports/{model.id}/{stamp}.json", "html": f"reports/{model.id}/{stamp}.html"}


HANDLERS: dict[str, Callable[[WorkerContext, Job, Path], dict[str, Any]]] = {
    "train": _train,
    "audit": _audit,
    "remediate": _remediate,
    "report": _report,
}
RECOVERY = {"audit": State.REGISTERED, "remediate": State.FINDINGS}  # state after a final failure


def _recover(ctx: WorkerContext, job: Job) -> None:
    """After the last attempt fails, return the version to a state a person can act on."""
    recovery = RECOVERY.get(job.kind)
    if recovery is None or "version" not in job.payload:
        return
    with scoped_session(ctx.factory, job.tenant_id) as session:
        version = session.get(ModelVersion, job.payload["version"])
        if version is not None and (version.state, recovery) in lifecycle.TRANSITIONS:
            lifecycle.move(session, version, recovery, WORKER, f"{job.kind} job failed")


def run_once(ctx: WorkerContext) -> bool:
    """Process one job. Returns False when the queue is empty."""
    with scoped_session(ctx.factory, ALL_TENANTS) as session:
        claimed = jobs.claim(session, ctx.name, tuple(HANDLERS))
        if claimed is None:
            return False
        session.expunge(claimed)
    try:
        with tempfile.TemporaryDirectory(prefix="mp-job-") as tmp:
            result = HANDLERS[claimed.kind](ctx, claimed, Path(tmp))
    except Exception as exc:  # noqa: BLE001 - any failure is recorded on the job
        message = f"{type(exc).__name__}: {exc}"
        with scoped_session(ctx.factory, ALL_TENANTS) as session:
            job = session.get(Job, claimed.id)
            assert job is not None
            jobs.fail(session, job, message + "\n" + traceback.format_exc(limit=3))
            final = job.status.value == "failed"
        if final:
            _recover(ctx, claimed)
        return True
    with scoped_session(ctx.factory, ALL_TENANTS) as session:
        job = session.get(Job, claimed.id)
        assert job is not None
        jobs.finish(session, job, result)
    return True


def drain(ctx: WorkerContext, limit: int = 100) -> int:
    """Process jobs until the queue is empty (or ``limit``); returns how many ran."""
    count = 0
    while count < limit and run_once(ctx):
        count += 1
    return count
