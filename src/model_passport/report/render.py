"""Render a passport as a self-contained HTML report (no template engine).

Each section is a small typed function returning ``Html``; ``report.html.el`` escapes every
value that is not already ``Html``, so passport contents cannot inject markup.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from model_passport.core.schema import (
    Finding,
    LeakageResult,
    Passport,
    PrivacyReport,
    RevisionInfo,
    SecurityReport,
    Severity,
    at_least,
)
from model_passport.core.verifier import VerificationReport
from model_passport.report.html import Child, Html, el, join, table
from model_passport.report.summary import Summary, summarize

CSS = resources.files("model_passport.report").joinpath("style.css").read_text(encoding="utf-8")


# --- Formatting ----------------------------------------------------------------------------


def short(value: object, n: int = 12) -> str:
    return f"{str(value)[:n]}…" if value else ""


def pct(value: float | None) -> str:
    return "" if value is None else f"{value:.1%}"


def num(value: float | None) -> str:
    return "" if value is None else f"{value:,.4g}"


def dash(value: object) -> object:
    return "–" if value is None or value == "" else value


def format_threshold(value: object) -> str:
    """``{"warn": 0.55, "fail": 0.6}`` -> ``warn 0.55 · fail 0.6``."""
    if isinstance(value, dict):
        return " · ".join(f"{level} {limit}" for level, limit in value.items())
    return str(value)


def kv(mapping: dict[str, object]) -> str:
    return ", ".join(f"{k}={v}" for k, v in mapping.items())


def badge(text: str, css: str) -> Html:
    return el("span", text, class_=f"badge {css}")


def stat(label: str, value: object) -> Html:
    return el("div", el("div", label, class_="k"), el("div", value, class_="v"), class_="stat")


def card(section_id: str, title: Child, *children: Child) -> Html:
    return el("section", el("h2", title), *children, class_="card", id=section_id)


def bullets(items: list[str], css: str = "points") -> Html:
    return el("ul", [el("li", i) for i in items], class_=css)


def mono(value: object) -> Html:
    return el("span", value, class_="mono")


# --- Sections ------------------------------------------------------------------------------


def _header(p: Passport, verdict: str) -> Html:
    built = p.identity.created_at.strftime("%Y-%m-%d %H:%M UTC")
    return el(
        "header",
        el(
            "div",
            el("div", "Model Passport", class_="sub"),
            el(
                "h1", p.identity.model_name, " ", el("span", f"v{p.identity.version}", class_="sub")
            ),
            el("div", f"{p.identity.passport_id} · built {built}", class_="sub mono"),
        ),
        badge(verdict if verdict != "none" else "no policy", f"big {verdict}"),
    )


def _summary(p: Passport, summary: Summary) -> Html:
    declared = p.declared
    lists = [
        ("Out of scope", declared.out_of_scope_uses),
        ("Known limitations", declared.known_limitations),
        ("Ethical risks", declared.ethical_risks),
    ]
    owner = None
    if declared.owner or declared.contact:
        contact = f" · {declared.contact}" if declared.contact else ""
        owner = el("p", f"Owner: {declared.owner or 'n/a'}{contact}", class_="sub")
    return card(
        "summary",
        summary.headline,
        el("p", el("b", "Intended use:"), " ", declared.intended_use)
        if declared.intended_use
        else None,
        [el("h3", "Concerns"), bullets(summary.concerns, "points concerns")]
        if summary.concerns
        else None,
        [el("h3", "Checks"), bullets(summary.points)] if summary.points else None,
        el(
            "div",
            [el("div", el("h3", title), bullets(items)) for title, items in lists if items],
            class_="grid",
        )
        if any(items for _, items in lists)
        else None,
        owner,
    )


def _verification(v: VerificationReport) -> Html:
    artifacts = (
        f"{len(v.artifacts) - len(v.changed)}/{len(v.artifacts)}"
        if v.artifacts
        else "not checked here"
    )
    return card(
        "verification",
        ["Verification ", badge("verified" if v.ok else "failed", "pass" if v.ok else "fail")],
        el(
            "div",
            stat("Signature", "valid" if v.signature_ok else "INVALID"),
            stat("Merkle root", "matches" if v.merkle_ok else "MISMATCH"),
            stat("Signing key", "matches" if v.fingerprint_ok else "MISMATCH"),
            stat("Artifacts unchanged", artifacts),
            stat("Lifecycle events", "chain intact" if not v.event_errors else "BROKEN"),
            class_="grid",
        ),
        [el("p", f"CHANGED {c.path} ({c.status.value})", class_="fail mono") for c in v.changed],
        [el("p", e, class_="fail mono") for e in v.event_errors],
    )


def _policy(p: Passport) -> Html | None:
    if p.policy is None:
        return None
    rows = [
        [mono(r.name), badge(r.result.value, r.result.value), dash(r.observed),
         mono(format_threshold(r.threshold)), r.message]
        for r in p.policy.rules
    ]  # fmt: skip
    return card(
        "policy",
        "Policy gate",
        table(["Rule", "Result", "Observed", "Threshold", "Detail"], rows),
        el("p", f"policy sha256 {short(p.policy.policy_sha256, 16)}", class_="sub mono"),
    )


def _model(p: Passport) -> Html:
    parts: list[Child] = []
    if p.model is not None:
        m = p.model
        parts += [
            el(
                "div",
                stat("Algorithm", dash(m.algorithm)),
                stat("Framework", dash(m.framework)),
                stat("Task", dash(m.task_type)),
                class_="grid",
            ),
            [el("h3", "Hyperparameters"), el("p", kv(m.hyperparameters), class_="mono")]
            if m.hyperparameters
            else None,
            el("p", f"{m.artifact_uri} · sha256 {short(m.artifact_sha256, 16)}", class_="sub mono"),
        ]
    if p.metrics:
        names = list(dict.fromkeys(n for values in p.metrics.values() for n in values))
        rows = [
            [split, *[num(values[n]) if n in values else "–" for n in names]]
            for split, values in p.metrics.items()
        ]
        parts += [el("h3", "Metrics"), table(["Split", *names], rows)]
    return card("model", "Model", parts)


def _leakage(leakage: LeakageResult) -> Html:
    return join(
        el("h3", "Membership inference (loss-threshold attack)"),
        el(
            "div",
            stat("Attack AUC (0.5 = no leakage)", f"{leakage.mia_auc:.3f}"),
            stat(f"TPR at {pct(leakage.low_fpr)} FPR", pct(leakage.tpr_at_low_fpr)),
            stat(
                f"Generalization gap ({leakage.gap_metric})",
                f"{leakage.generalization_gap or 0:.3f}",
            ),
            stat("Members / non-members", f"{leakage.members} / {leakage.nonmembers}"),
            class_="grid",
        ),
    )


def _severity_badge(finding: Finding) -> Html:
    return badge(finding.severity.value, finding.severity.value)


def _privacy(pr: PrivacyReport) -> Html:
    reid_rows = [
        [r.dataset, mono(", ".join(r.quasi_identifiers) or "–"), dash(r.k_anonymity),
         dash(r.l_diversity), pct(r.unique_fraction),
         mono("; ".join(f"{'+'.join(c.columns)} ({pct(c.unique_fraction)})"
                        for c in r.risky_combinations[:4]))]
        for r in pr.reidentification
    ]  # fmt: skip
    pii = [f for f in pr.data_findings if f.scanner == "pii"]
    pii_rows = [
        [_severity_badge(f), mono(f.category), mono(f.location), f.count,
         pct(f.details["hit_rate"]) if f.details.get("hit_rate") is not None else "name only",
         mono(", ".join(f.masked_examples))]
        for f in pii
    ]  # fmt: skip
    scanned = ", ".join(pr.datasets_scanned) or "none scanned"
    return card(
        "privacy",
        "Privacy",
        _leakage(pr.leakage) if pr.leakage else None,
        [
            el("h3", "Reidentification risk"),
            table(
                [
                    "Dataset",
                    "Quasi-identifiers",
                    "k-anonymity",
                    "l-diversity",
                    "Unique records",
                    "Risky combinations",
                ],
                reid_rows,
            ),
        ]
        if reid_rows
        else None,
        el("h3", f"Data scan findings ({scanned})"),
        table(["Severity", "Entity", "Location", "Rows", "Hit rate", "Masked examples"], pii_rows)
        if pii_rows
        else el("p", "No PII detected."),
    )


def _security(sr: SecurityReport) -> Html:
    confirmed = sum(1 for f in sr.secret_findings if at_least(f.severity, Severity.HIGH))
    findings = sr.secret_findings + sr.artifact_findings + sr.dependency_vulnerabilities
    rows = [
        [_severity_badge(f), f.scanner, mono(f.category), mono(f.location), f.message]
        for f in findings
    ]
    return card(
        "security",
        "Security",
        el(
            "div",
            stat("Files scanned for secrets", dash(sr.files_scanned_for_secrets)),
            stat("Confirmed secrets", confirmed),
            stat("Artifacts scanned", len(sr.artifacts_scanned)),
            stat(
                f"Dependency CVEs ({sr.dependency_audit.value})",
                len(sr.dependency_vulnerabilities),
            ),
            class_="grid",
        ),
        table(["Severity", "Check", "Category", "Location", "Detail"], rows) if rows else None,
        el("p", sr.dependency_audit_message, class_="sub") if sr.dependency_audit_message else None,
    )


def _pipeline(p: Passport) -> Html | None:
    if not p.pipeline:
        return None
    stages: list[Child] = []
    for index, s in enumerate(p.pipeline, start=1):
        git = f"{short(s.git_commit, 8) or 'no git'}{' (dirty)' if s.git_dirty else ''}"
        stages.append(
            el(
                "div",
                el("b", f"{index}. {s.name}"),
                el("div", f"{s.script_path} @ {git}", class_="mono"),
                el("div", kv(s.parameters), class_="mono sub") if s.parameters else None,
                el("div", "in: ", mono(", ".join(i.path for i in s.inputs) or "–"), class_="sub"),
                el("div", "out: ", mono(", ".join(o.path for o in s.outputs) or "–"), class_="sub"),
                class_="stage",
            )
        )
        if index < len(p.pipeline):
            stages.append(el("div", "→", class_="arrow"))
    return card("pipeline", "Pipeline", el("div", stages, class_="flow"))


def _datasets(p: Passport) -> Html | None:
    if not p.datasets:
        return None
    rows = [
        [d.name, d.split_role.value, f"{d.row_count:,}" if d.row_count is not None else "–",
         len(d.column_schema) if d.column_schema else "–", dash(d.source), dash(d.license),
         mono(short(d.sha256))]
        for d in p.datasets
    ]  # fmt: skip
    return card(
        "datasets",
        "Datasets",
        table(["Name", "Role", "Rows", "Columns", "Source", "License", "SHA256"], rows),
    )


def _revision(rv: RevisionInfo) -> Html:
    rows = [
        [c.name, c.status.value, dash(c.previous_rows), dash(c.current_rows)]
        for c in rv.dataset_changes
    ]
    deltas = "; ".join(
        f"{split}.{name} {delta:+.4f}"
        for split, values in rv.metric_deltas.items()
        for name, delta in values.items()
    )
    counts = (
        f"{len(rv.changed_artifacts)} changed, {len(rv.added_artifacts)} added, "
        f"{len(rv.removed_artifacts)} removed artifacts · supersedes "
    )
    return card(
        "revision",
        f"Changes since v{rv.previous_version}",
        el("p", rv.reason) if rv.reason else None,
        table(["Dataset", "Status", "Rows before", "Rows now"], rows),
        [el("h3", "Metric changes"), el("p", deltas, class_="mono")] if deltas else None,
        el("p", counts, mono(rv.previous_passport_id), class_="sub"),
    )


def _events(p: Passport) -> Html | None:
    if not p.events:
        return None
    rows = []
    for e in p.events:
        if e.event_type == "drift_check":
            drifted = ", ".join(e.payload.get("drifted_features", []))
            state = (
                f"drift detected in {drifted}" if e.payload.get("drift_detected") else "no drift"
            )
            summary = f"{state} ({e.payload.get('rows')} rows)"
        else:
            summary = str(e.payload.get("summary", ""))
        rows.append([mono(e.timestamp.strftime("%Y-%m-%d %H:%M")), e.event_type, summary])
    return card("events", "Lifecycle events", table(["When", "Type", "Summary"], rows))


def _identity(p: Passport) -> Html:
    ident = p.identity
    rows: list[list[Child]] = [
        [el("b", "Merkle root"), mono(ident.merkle_root)],
        [el("b", "Signing key"), mono(ident.public_key_fingerprint)],
        [el("b", "Signature"), mono(short(ident.signature, 32))],
    ]
    if p.lineage_links:
        rows.append([el("b", "Upstream models"), mono(", ".join(map(str, p.lineage_links)))])
    artifact_rows = [
        [mono(a.path), a.kind.value, f"{a.size_bytes:,} B", mono(short(a.sha256, 16))]
        for a in p.artifacts
    ]
    env = p.environment
    return card(
        "identity",
        "Identity",
        el("table", [el("tr", [el("td", cell) for cell in row]) for row in rows]),
        el(
            "details",
            el("summary", f"{len(p.artifacts)} hashed artifacts"),
            el("table", [el("tr", [el("td", c) for c in row]) for row in artifact_rows]),
        ),
        el(
            "details",
            el(
                "summary",
                f"Environment: Python {env.python_version} on {env.os} "
                f"({len(env.dependencies)} packages)",
            ),
            el("p", ", ".join(f"{k}=={v}" for k, v in env.dependencies.items()), class_="mono"),
        )
        if env
        else None,
    )


# --- Document ------------------------------------------------------------------------------


def render_html(passport: Passport, verification: VerificationReport | None = None) -> str:
    summary = summarize(passport)
    verdict = summary.verdict.value if summary.verdict else "none"
    body = el(
        "main",
        _header(passport, verdict),
        _summary(passport, summary),
        _verification(verification) if verification else None,
        _policy(passport),
        _model(passport),
        _privacy(passport.privacy_report) if passport.privacy_report else None,
        _security(passport.security_report) if passport.security_report else None,
        _pipeline(passport),
        _datasets(passport),
        _revision(passport.revision) if passport.revision else None,
        _events(passport),
        _identity(passport),
        el(
            "footer",
            "Generated by Model Passport. Verify with ",
            el("code", "passport verify"),
            "; this page is a view of the signed passport.json, not a substitute for it.",
        ),
    )
    title = f"Model Passport: {passport.identity.model_name} {passport.identity.version}"
    head = el(
        "head",
        Html('<meta charset="utf-8">'),
        Html('<meta name="viewport" content="width=device-width, initial-scale=1">'),
        el("title", title),
        el("style", Html(CSS)),
    )
    return f"<!doctype html>\n{el('html', head, el('body', body), lang='en')}\n"


def write_html(
    passport: Passport, out: Path, verification: VerificationReport | None = None
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(passport, verification), encoding="utf-8")
