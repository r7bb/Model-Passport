"""Remove the risk found by an audit from the training data (the sanitize step of remediation).

Strategies:

- ``surrogate`` (default): replace each risky value with a realistic, same-type synthetic
  value. The same original always maps to the same surrogate, so the text stays coherent
  and useful for training (pseudonymization).
- ``mask``: replace it with a type placeholder such as ``[EMAIL]``.
- ``drop``: remove every record that contains a risky value.

Values are chosen by severity from the audit findings, plus any entity types that should never
be trained on (data minimization, e.g. ``SSN`` and ``CREDITCARDNUMBER``) whatever their score.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from model_passport.core.schema import EntityAuditResult, Severity, at_least
from model_passport.llm.entities import Record, Span
from model_passport.llm.references import ReferenceSampler

STRATEGIES = ("surrogate", "mask", "drop")


@dataclass
class SanitizeReport:
    strategy: str
    replaced: Counter[str] = field(default_factory=Counter)
    dropped_records: int = 0

    def describe(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "replaced": dict(self.replaced),
            "dropped_records": self.dropped_records,
        }


def risky_spans(
    audit: EntityAuditResult | None,
    records: Iterable[Record],
    min_severity: Severity = Severity.MEDIUM,
    always: Iterable[str] = (),
) -> dict[str, set[tuple[int, int]]]:
    """Record id -> spans to sanitize: audited findings at ``min_severity`` or above, plus all
    entities of the ``always`` types."""
    chosen: dict[str, set[tuple[int, int]]] = {}
    if audit is not None:
        for finding in audit.findings:
            if at_least(finding.severity, min_severity):
                chosen.setdefault(finding.record, set()).add(tuple(finding.span))  # type: ignore[arg-type]
    kinds = {k.upper() for k in always}
    if kinds:
        for record in records:
            for span in record.entities:
                if span.entity_type in kinds:
                    chosen.setdefault(record.id, set()).add((span.start, span.end))
    return chosen


def _rewrite(
    record: Record, targets: set[tuple[int, int]], replace: dict[str, str], strategy: str,
    sampler: ReferenceSampler, report: SanitizeReport,
) -> Record:  # fmt: skip
    """Rebuild the text left to right, shifting later spans by each replacement's length."""
    text = record.text
    spans = sorted(record.entities, key=lambda s: s.start)
    new_spans: list[Span] = []
    shift = 0
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        pieces.append(text[cursor : span.start])
        value = record.value(span)
        if (span.start, span.end) in targets:
            key = f"{span.entity_type}\x00{value}"
            if key not in replace:
                replace[key] = (
                    f"[{span.entity_type}]"
                    if strategy == "mask"
                    else sampler.one(span.entity_type, value)
                )
            value = replace[key]
            report.replaced[span.entity_type] += 1
        start = span.start + shift
        pieces.append(value)
        new_spans.append(Span(start, start + len(value), span.entity_type))
        shift += len(value) - (span.end - span.start)
        cursor = span.end
    pieces.append(text[cursor:])
    return Record(record.id, "".join(pieces), new_spans)


def sanitize(
    records: Sequence[Record],
    targets: dict[str, set[tuple[int, int]]],
    strategy: str = "surrogate",
    seed: int = 0,
) -> tuple[list[Record], SanitizeReport]:
    """A sanitized copy of ``records`` and what was changed."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}")
    report = SanitizeReport(strategy)
    if strategy == "drop":
        kept = [r for r in records if not targets.get(r.id)]
        report.dropped_records = len(records) - len(kept)
        return kept, report
    sampler = ReferenceSampler(records, source="synthetic", seed=seed)
    replace: dict[str, str] = {}
    out = []
    for record in records:
        wanted = targets.get(record.id, set())
        out.append(
            _rewrite(record, wanted, replace, strategy, sampler, report) if wanted else record
        )
    return out, report
