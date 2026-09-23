"""Graphviz DOT for a passport's pipeline DAG and lineage."""

from __future__ import annotations

from model_passport.core.schema import ArtifactKind, Passport

_COLORS = {
    ArtifactKind.DATASET: "#dbeafe",
    ArtifactKind.MODEL: "#dcfce7",
    ArtifactKind.SCRIPT: "#f3f4f6",
    ArtifactKind.CONFIG: "#f3f4f6",
    ArtifactKind.OTHER: "#fef9c3",
}


def _quote(text: str) -> str:
    """A DOT string literal; real newlines become line breaks in the rendered label."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def pipeline_dot(passport: Passport) -> str:
    """Artifacts as ellipses, stages as boxes, edges input -> stage -> output."""
    kinds = {a.path: a.kind for a in passport.artifacts}
    lines = [
        "digraph pipeline {",
        '  rankdir=LR; bgcolor="transparent";',
        '  node [fontname="Helvetica", fontsize=11];',
    ]
    paths: set[str] = set()
    for stage in passport.pipeline:
        paths.update(i.path for i in stage.inputs)
        paths.update(o.path for o in stage.outputs)
    for path in sorted(paths):
        color = _COLORS.get(kinds.get(path, ArtifactKind.OTHER), "#ffffff")
        lines.append(f'  {_quote(path)} [shape=ellipse, style=filled, fillcolor="{color}"];')
    for stage in passport.pipeline:
        node = _quote(f"stage:{stage.name}")
        lines.append(
            f'  {node} [label={_quote(stage.name)}, shape=box, style="rounded,filled", '
            'fillcolor="#1f2937", fontcolor="white"];'
        )
        lines.extend(f"  {_quote(i.path)} -> {node};" for i in stage.inputs)
        lines.extend(f"  {node} -> {_quote(o.path)};" for o in stage.outputs)
    lines.append("}")
    return "\n".join(lines)


def lineage_dot(passport: Passport, names: dict[str, str] | None = None) -> str:
    """This passport, its upstream models, and the version it supersedes."""
    names = names or {}
    me = str(passport.identity.passport_id)
    label = f"{passport.identity.model_name}\nv{passport.identity.version}"
    lines = [
        "digraph lineage {",
        '  rankdir=LR; bgcolor="transparent";',
        '  node [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled"];',
        f'  {_quote(me)} [label={_quote(label)}, fillcolor="#dcfce7"];',
    ]
    for link in passport.lineage_links:
        uid = str(link)
        lines.append(
            f'  {_quote(uid)} [label={_quote(names.get(uid, uid[:8]))}, fillcolor="#e5e7eb"];'
        )
        lines.append(f'  {_quote(uid)} -> {_quote(me)} [label="upstream"];')
    revision = passport.revision
    if revision is not None:
        uid = str(revision.previous_passport_id)
        default = f"{passport.identity.model_name}\nv{revision.previous_version}"
        prev_label = names.get(uid, default)
        lines.append(f'  {_quote(uid)} [label={_quote(prev_label)}, fillcolor="#f3f4f6"];')
        lines.append(f'  {_quote(uid)} -> {_quote(me)} [label="superseded by", style=dashed];')
    lines.append("}")
    return "\n".join(lines)
