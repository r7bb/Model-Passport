"""The diligence report (M9) for investors, acquirers, and auditors.

It answers the questions data-provenance diligence asks: which data trained which model
version, where that data came from and under what license and consent, what personal data the
model memorized, what was done about it, who approved each release, and whether the record can
be trusted. Every version's signed attestation is re-verified against the tenant's public key,
and the audit log's hash chain is checked, when the report is generated.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from model_passport.platform import attest, auditlog
from model_passport.platform.models import Approval, Dataset, Model, ModelVersion, Tenant, now
from model_passport.report.html import Html, el, join, table

STYLE = """
body{font:15px/1.5 -apple-system,Segoe UI,sans-serif;margin:2rem auto;max-width:960px;color:#111}
h1{margin-bottom:.2rem}h2{margin-top:2rem;border-bottom:1px solid #ddd}
table{border-collapse:collapse;width:100%;margin:.5rem 0}
th,td{border:1px solid #ddd;padding:.35rem .5rem;text-align:left;font-size:13px}
th{background:#f4f4f5}.ok{color:#15803d}.bad{color:#b91c1c}.muted{color:#555}
"""


def _dataset(ds: Dataset | None) -> dict[str, Any] | None:
    if ds is None:
        return None
    return {
        "id": ds.id,
        "name": ds.name,
        "sha256": ds.sha256,
        "records": ds.records,
        "entities": ds.entities,
        "source": ds.source,
        "license": ds.license,
        "consent": ds.consent,
        "added": ds.created_at.isoformat(),
    }


def _version(session: Session, tenant: Tenant, version: ModelVersion) -> dict[str, Any]:
    approvals = session.scalars(select(Approval).where(Approval.version_id == version.id)).all()
    verified = (
        attest.verify(version.attestation, tenant.public_key)
        if version.attestation and tenant.public_key
        else None
    )
    parent = session.get(ModelVersion, version.parent_id) if version.parent_id else None
    return {
        "version": version.version,
        "state": version.state.value,
        "parent": parent.version if parent else None,
        "training_data": version.dataset_id,
        "reference_data": version.reference_dataset_id,
        "verdict": version.verdict,
        "auc": version.auc,
        "critical": version.critical,
        "high": version.high,
        "attestation_sha256": version.attestation_sha256,
        "attestation_verified": verified,
        "approvals": [
            {
                "decision": a.decision,
                "role": a.role.value,
                "comment": a.comment,
                "at": a.created_at.isoformat(),
            }
            for a in approvals
        ],
        "created": version.created_at.isoformat(),
    }


def diligence(session: Session, tenant: Tenant, model: Model) -> dict[str, Any]:
    """The report as data (JSON-serializable)."""
    versions = list(model.versions)
    dataset_ids = {i for v in versions for i in (v.dataset_id, v.reference_dataset_id) if i}
    datasets = [_dataset(session.get(Dataset, i)) for i in sorted(dataset_ids)]
    events = auditlog.chain(session, tenant.id)
    problems = auditlog.verify_chain(events)
    rows = [_version(session, tenant, v) for v in versions]
    signed = [r["attestation_verified"] for r in rows if r["attestation_sha256"]]
    no_consent = [d["name"] for d in datasets if d and d["consent"] == "unknown"]
    return {
        "type": "mp.diligence-report/v1",
        "generated_at": now().isoformat(),
        "tenant": {"slug": tenant.slug, "name": tenant.name},
        "model": {
            "name": model.name,
            "access": model.access.value,
            "base": model.base,
            "description": model.description,
        },
        "summary": {
            "versions": len(rows),
            "current": rows[-1] if rows else None,
            "all_attestations_verified": all(signed),
            "datasets_without_consent": no_consent,
        },
        "versions": rows,
        "datasets": datasets,
        "audit_log": {
            "events": len(events),
            "intact": not problems,
            "problems": problems,
            "head": {"seq": events[-1].seq, "hash": events[-1].hash} if events else None,
        },
        "public_key": tenant.public_key,
    }


def _yes(value: object) -> Html:
    if value is None:
        return el("span", "n/a", class_="muted")
    return el("span", "verified" if value else "FAILED", class_="ok" if value else "bad")


def render_html(report: dict[str, Any]) -> str:
    """A self-contained HTML page for the report."""
    model, summary, log = report["model"], report["summary"], report["audit_log"]
    versions = table(
        ["Version", "State", "Trained on", "Verdict", "AUC", "Critical", "High", "Attestation"],
        [
            [
                v["version"],
                v["state"],
                v["training_data"] or "-",
                v["verdict"] or "-",
                "-" if v["auc"] is None else f"{v['auc']:.3f}",
                v["critical"],
                v["high"],
                _yes(v["attestation_verified"]),
            ]
            for v in report["versions"]
        ],
    )
    datasets = table(
        ["Dataset", "SHA256", "Records", "Source", "License", "Consent"],
        [
            [
                d["name"],
                d["sha256"][:16] + "…",
                d["records"],
                d["source"],
                d["license"],
                d["consent"],
            ]
            for d in report["datasets"]
            if d
        ],
    )
    body = join(
        el("h1", f"Diligence report: {model['name']}"),
        el("p", f"{report['tenant']['name']} · generated {report['generated_at']}", class_="muted"),
        el("h2", "Summary"),
        el(
            "ul",
            el("li", f"{summary['versions']} version(s); model access: {model['access']}"),
            el("li", "All attestations verified: ", _yes(summary["all_attestations_verified"])),
            el("li", "Audit log: ", _yes(log["intact"]), f" ({log['events']} events)"),
            el(
                "li",
                "Datasets with unknown consent: ",
                ", ".join(summary["datasets_without_consent"]) or "none",
            ),
        ),
        el("h2", "Versions and lineage"),
        versions,
        el("h2", "Training data provenance"),
        datasets,
        el("h2", "Verification"),
        el("p", "Each attestation is signed with the organization's Ed25519 key:"),
        el("pre", report["public_key"] or "not published"),
    )
    head = join(
        el("meta", charset="utf-8"),
        el("title", f"Diligence report: {model['name']}"),
        el("style", Html(STYLE)),
    )
    return "<!doctype html>" + el("html", el("head", head), el("body", body))
