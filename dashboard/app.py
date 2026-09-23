"""Model Passport dashboard.

    streamlit run dashboard/app.py                      # reads ./passport.json and history
    PASSPORT_REGISTRY_URL=http://localhost:8000 streamlit run dashboard/app.py

Deep link to a version with ``?passport=<passport_id>``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from model_passport.core.schema import LifecycleEvent, Passport
from model_passport.registry.client import RegistryClient, RegistryError
from model_passport.registry.sources import Entry, LocalSource, PassportSource, RegistrySource
from model_passport.report.dag import lineage_dot, pipeline_dot
from model_passport.report.summary import summarize

VERDICT_STYLE = {
    "pass": ("#15803d", "#dcfce7"),
    "warn": ("#b45309", "#fef3c7"),
    "fail": ("#b91c1c", "#fee2e2"),
    None: ("#374151", "#e5e7eb"),
}
SOURCES = ["Local files", "Registry"]
TABS = ["Simple view", "Policy", "Pipeline", "Lineage", "Privacy", "Security", "Monitoring",
        "Raw JSON"]  # fmt: skip

st.set_page_config(page_title="Model Passport", page_icon="🛂", layout="wide")


def badge(verdict: str | None, size: str = "15px") -> str:
    fg, bg = VERDICT_STYLE.get(verdict, VERDICT_STYLE[None])
    text = (verdict or "no policy").upper()
    return (
        f"<span style='background:{bg};color:{fg};padding:4px 14px;border-radius:999px;"
        f"font-weight:700;font-size:{size};letter-spacing:.05em'>{text}</span>"
    )


def table(rows: list[dict[str, Any]]) -> None:
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def short(value: str | None, n: int = 12) -> str:
    return f"{value[:n]}…" if value else ""


# --- Source and selection ------------------------------------------------------------------


def make_source() -> PassportSource | None:
    st.sidebar.title("🛂 Model Passport")
    registry_url = os.environ.get("PASSPORT_REGISTRY_URL")
    kind = st.sidebar.radio("Source", SOURCES, index=1 if registry_url else 0)
    if kind == "Registry":
        url = st.sidebar.text_input("Registry URL", registry_url or "http://localhost:8000")
        return RegistrySource(RegistryClient(url))
    root = st.sidebar.text_input("Project directory", os.environ.get("PASSPORT_PROJECT_DIR", "."))
    if not Path(root).is_dir():
        st.sidebar.error("Directory not found")
        return None
    return LocalSource(Path(root))


def pick(entries: list[Entry]) -> Entry | None:
    wanted = st.query_params.get("passport")
    linked = next((e for e in entries if e.passport_id == wanted), None)
    models = sorted({e.model_name for e in entries})
    model = st.sidebar.selectbox(
        "Model", models, index=models.index(linked.model_name) if linked else 0
    )
    versions = [e for e in entries if e.model_name == model]
    index = versions.index(linked) if linked in versions else 0
    return st.sidebar.selectbox("Version", versions, index=index, format_func=lambda e: e.label)


# --- Header and tabs -----------------------------------------------------------------------


def header(p: Passport, verification: dict[str, Any]) -> None:
    left, right = st.columns([4, 1])
    left.markdown(
        f"## {p.identity.model_name} <span style='color:#6b7280'>v{p.identity.version}</span>",
        unsafe_allow_html=True,
    )
    left.caption(f"`{p.identity.passport_id}` · built {p.identity.created_at:%Y-%m-%d %H:%M UTC}")
    verdict = p.policy.verdict.value if p.policy else None
    right.markdown(
        f"<div style='text-align:right;margin-top:18px'>{badge(verdict, '20px')}</div>",
        unsafe_allow_html=True,
    )

    privacy = p.privacy_report
    leakage = privacy.leakage if privacy else None
    ks = [r.k_anonymity for r in (privacy.reidentification if privacy else []) if r.k_anonymity]
    accuracy = p.metrics.get("test", {}).get("accuracy")
    cols = st.columns(5)
    cols[0].metric("Signature", "valid" if verification.get("ok") else "INVALID")
    cols[1].metric("Test accuracy", f"{accuracy:.3f}" if accuracy is not None else "–")
    cols[2].metric(
        "Attack AUC",
        f"{leakage.mia_auc:.3f}" if leakage else "–",
        help="Membership inference; 0.5 means no leakage",
    )
    cols[3].metric("k-anonymity", min(ks) if ks else "–")
    cols[4].metric("Lifecycle events", len(p.events))


def simple_view(p: Passport) -> None:
    summary = summarize(p)
    st.markdown(f"### {summary.headline}")
    if p.declared.intended_use:
        st.info(f"**Intended use:** {p.declared.intended_use}")
    for concern in summary.concerns:
        st.error(concern, icon="⚠️")
    for point in summary.points:
        st.success(point, icon="✅")
    sections = [
        ("Out of scope", p.declared.out_of_scope_uses),
        ("Known limitations", p.declared.known_limitations),
        ("Ethical risks", p.declared.ethical_risks),
    ]
    for col, (title, items) in zip(st.columns(3), sections, strict=True):
        col.markdown(f"**{title}**")
        for item in items or ["–"]:
            col.markdown(f"- {item}")


def policy_view(p: Passport) -> None:
    if p.policy is None:
        st.warning("No policy was applied.")
        return
    table(
        [
            {"rule": r.name, "result": r.result.value.upper(), "observed": r.observed,
             "threshold": str(r.threshold), "detail": r.message}
            for r in p.policy.rules
        ]
    )  # fmt: skip
    st.caption(f"policy sha256 `{p.policy.policy_sha256}`")


def pipeline_view(p: Passport) -> None:
    if not p.pipeline:
        st.info("No recorded pipeline (passport built from manual inputs).")
        return
    st.graphviz_chart(pipeline_dot(p), width="stretch")
    rows = []
    for s in p.pipeline:
        seconds = (
            (s.ended_at - s.started_at).total_seconds() if s.started_at and s.ended_at else None
        )
        rows.append(
            {"stage": s.name, "script": s.script_path,
             "git": short(s.git_commit, 8) + (" (dirty)" if s.git_dirty else ""),
             "parameters": ", ".join(f"{k}={v}" for k, v in s.parameters.items()),
             "seconds": round(seconds, 2) if seconds is not None else None}
        )  # fmt: skip
    table(rows)


def lineage_view(p: Passport, entries: list[Entry]) -> None:
    names = {e.passport_id: f"{e.model_name}\nv{e.version} {(e.verdict or '').upper()}"
             for e in entries}  # fmt: skip
    st.graphviz_chart(lineage_dot(p, names), width="content")
    rv = p.revision
    if rv is None:
        st.caption("First version of this model: nothing superseded.")
        return
    st.markdown(f"**Revision {rv.sequence}** · supersedes `{short(str(rv.previous_passport_id))}`")
    if rv.reason:
        st.write(rv.reason)
    table(
        [
            {"dataset": c.name, "status": c.status.value, "rows before": c.previous_rows,
             "rows now": c.current_rows, "previous sha256": short(c.previous_sha256),
             "current sha256": short(c.current_sha256)}
            for c in rv.dataset_changes
        ]
    )  # fmt: skip
    deltas = [
        {"split": split, "metric": name, "change": f"{delta:+.4f}"}
        for split, values in rv.metric_deltas.items()
        for name, delta in values.items()
    ]
    if deltas:
        st.markdown("**Metric changes since the previous version**")
        table(deltas)


def privacy_view(p: Passport) -> None:
    pr = p.privacy_report
    if pr is None:
        st.info("No privacy report.")
        return
    if pr.leakage:
        cols = st.columns(4)
        cols[0].metric("Attack AUC", f"{pr.leakage.mia_auc:.3f}")
        cols[1].metric(f"TPR @ {pr.leakage.low_fpr:.0%} FPR", f"{pr.leakage.tpr_at_low_fpr:.1%}")
        cols[2].metric("Generalization gap", f"{pr.leakage.generalization_gap or 0:+.3f}")
        cols[3].metric("Members / non-members", f"{pr.leakage.members} / {pr.leakage.nonmembers}")
    if pr.reidentification:
        st.markdown("**Reidentification risk**")
        table(
            [
                {"dataset": r.dataset, "quasi-identifiers": ", ".join(r.quasi_identifiers),
                 "k": r.k_anonymity, "l": r.l_diversity,
                 "unique %": round(r.unique_fraction * 100, 2),
                 "risky combinations": "; ".join("+".join(c.columns)
                                                 for c in r.risky_combinations[:4])}
                for r in pr.reidentification
            ]
        )  # fmt: skip
    st.markdown(f"**Data findings** (scanned: {', '.join(pr.datasets_scanned) or 'none'})")
    findings = [f.model_dump(mode="json") for f in pr.data_findings if f.scanner == "pii"]
    if findings:
        columns = ["severity", "category", "location", "count", "masked_examples"]
        st.dataframe(pd.DataFrame(findings)[columns], hide_index=True, width="stretch")
    else:
        st.success("No PII detected.")


def security_view(p: Passport) -> None:
    sr = p.security_report
    if sr is None:
        st.info("No security report.")
        return
    cols = st.columns(3)
    cols[0].metric("Files scanned for secrets", sr.files_scanned_for_secrets or 0)
    cols[1].metric("Artifacts scanned", len(sr.artifacts_scanned))
    cols[2].metric(
        f"Dependency CVEs ({sr.dependency_audit.value})", len(sr.dependency_vulnerabilities)
    )
    findings = sr.secret_findings + sr.artifact_findings + sr.dependency_vulnerabilities
    if findings:
        columns = ["severity", "scanner", "category", "location", "message"]
        frame = pd.DataFrame([f.model_dump(mode="json") for f in findings])[columns]
        st.dataframe(frame, hide_index=True, width="stretch")
    st.caption(sr.dependency_audit_message)


def _drift_row(e: LifecycleEvent) -> dict[str, Any]:
    payload = e.payload
    return {
        "time": e.timestamp,
        "batch": payload.get("batch"),
        "rows": payload.get("rows"),
        "drift": payload.get("drift_detected"),
        "drifted features": ", ".join(payload.get("drifted_features", [])),
        "live accuracy": (payload.get("performance") or {}).get("accuracy"),
        "retrain": payload.get("retrain_recommended"),
    }


def monitoring_view(p: Passport) -> None:
    drift = [e for e in p.events if e.event_type == "drift_check"]
    if not drift:
        st.info("No drift checks yet. Run `passport monitor drift <batch.csv>`.")
        return
    table([_drift_row(e) for e in drift])
    latest = drift[-1].payload.get("features", {})
    st.markdown(
        f"**Population stability index, latest batch** ({drift[-1].payload.get('batch')}; "
        "≥ 0.1 moderate, ≥ 0.25 major shift)"
    )
    st.bar_chart(pd.Series({k: v["psi"] for k, v in latest.items()}, name="PSI"))
    if len(drift) > 1:
        st.markdown("**PSI per feature over time**")
        history = pd.DataFrame(
            [
                {"time": e.timestamp,
                 **{name: f["psi"] for name, f in e.payload.get("features", {}).items()}}
                for e in drift
            ]
        ).set_index("time")  # fmt: skip
        st.line_chart(history)


def main() -> None:
    source = make_source()
    if source is None:
        return
    try:
        entries = source.entries()
    except RegistryError as exc:
        st.error(str(exc))
        return
    if not entries:
        st.info("No passports found. Run `passport build`, or `passport push` to a registry.")
        return
    entry = pick(entries)
    if entry is None:
        return
    st.query_params["passport"] = entry.passport_id
    document = source.document(entry.passport_id)
    passport = Passport.model_validate(document)
    verification = source.verification(entry.passport_id)
    header(passport, verification)

    views = [
        lambda: simple_view(passport),
        lambda: policy_view(passport),
        lambda: pipeline_view(passport),
        lambda: lineage_view(passport, entries),
        lambda: privacy_view(passport),
        lambda: security_view(passport),
        lambda: monitoring_view(passport),
        lambda: st.json(document, expanded=False),
    ]
    for tab, view in zip(st.tabs(TABS), views, strict=True):
        with tab:
            view()
    if verification.get("ok"):
        st.sidebar.success("Signature, Merkle root, and event chain verified")
    else:
        st.sidebar.error(f"Verification failed: {verification}")


main()
