"""The remediation loop: sanitize, retrain, re-audit, and version, until the gate passes.

Each round follows the product flow:

1. **Detect:** run the pipeline (fine-tune, audit) and build a signed passport. The passport
   links to the one it supersedes, and the old passport is archived, so every version is kept.
2. **Gate:** stop if the policy verdict is not fail and no confirmed High or Critical entity is
   left.
3. **Remediate:** archive the current training data under ``data/history/``, replace the
   risky values in ``data/train.jsonl``, bump the minor version (1.0.0 -> 1.1.0), and repeat.

Remediation starts entity by entity. Retraining can make other, previously borderline values of
the same type detectable, so if a type has confirmed High or Critical findings in two rounds in
a row, the next round replaces every value of that type (data minimization for that type).
Likewise, when the overall attack fails the gate but no single entity is flagged, the leakage is
spread across a type: every type whose own attack AUC is above ``type_auc_max`` is replaced.
There are only so many types, so the loop converges.

Audit findings locate entities in the original corpus (``data/raw.jsonl``). The training copy
has the same records and entities in the same order, but earlier replacements may have moved
their character offsets, so findings are translated by position in each record.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from model_passport.core.builder import build_passport, write_passport
from model_passport.core.capture import run_pipeline, save_run_record
from model_passport.core.config import load_config
from model_passport.core.revision import archive, load_previous
from model_passport.core.schema import EntityAuditResult, Passport, Severity, Verdict, at_least
from model_passport.llm.entities import Record, load_corpus, write_corpus
from model_passport.llm.project import HISTORY_DIR, RAW_CORPUS, TRAIN_CORPUS
from model_passport.llm.sanitize import risky_spans, sanitize
from model_passport.report.render import write_html

# ``version:`` inside the top-level ``project:`` block (other keys may come first).
VERSION_RE = re.compile(
    r"^(?P<head>project:\s*\n(?:[ \t]+.*\n)*?[ \t]+version:[ \t]*)(?P<version>\S+)", re.MULTILINE
)
Echo = Callable[[str], None]


class RemediationError(RuntimeError):
    """The project cannot be remediated as configured."""


@dataclass(frozen=True)
class RemediationSettings:
    max_rounds: int = 3
    min_severity: Severity = Severity.MEDIUM
    always: tuple[str, ...] = ("SSN", "CREDITCARDNUMBER")  # never train on these (minimization)
    strategy: str = "surrogate"
    seed: int = 0
    type_auc_max: float = 0.60  # the default el_mia_auc_max fail level


@dataclass
class Round:
    version: str
    passport_id: str
    verdict: str
    auc: float | None
    severity_counts: dict[str, int]
    risky_types: set[str] = field(default_factory=set)
    type_auc: dict[str, float] = field(default_factory=dict)
    sanitized: dict[str, int] = field(default_factory=dict)
    whole_types: set[str] = field(default_factory=set)

    @property
    def blocking(self) -> int:
        return self.severity_counts.get("critical", 0) + self.severity_counts.get("high", 0)


def bump_minor(version: str) -> str:
    """1.0.0 -> 1.1.0; anything not semver-like gets a ``.1`` suffix."""
    parts = version.split(".")
    if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]):
        return ".".join([parts[0], str(int(parts[1]) + 1), *(["0"] * max(len(parts) - 2, 1))])
    return f"{version}.1"


def set_version(config_path: Path, version: str) -> None:
    """Rewrite ``project.version`` in place, keeping the file's comments and layout."""
    text = config_path.read_text(encoding="utf-8")
    updated, count = VERSION_RE.subn(lambda m: f"{m['head']}{version}", text, count=1)
    if not count:
        raise RemediationError(f"no project version line in {config_path}")
    config_path.write_text(updated, encoding="utf-8")


def translate(
    targets: dict[str, set[tuple[int, int]]],
    raw: Sequence[Record],
    train: Sequence[Record],
    only_original: bool = False,
) -> dict[str, set[tuple[int, int]]]:
    """Map spans found in the original corpus to the current training copy, by position.

    With ``only_original``, values an earlier round already replaced are left alone.
    """
    current = {r.id: r for r in train}
    out: dict[str, set[tuple[int, int]]] = {}
    for record in raw:
        wanted = targets.get(record.id)
        copy = current.get(record.id)
        if not wanted or copy is None or len(copy.entities) != len(record.entities):
            continue
        for original, now in zip(record.entities, copy.entities, strict=True):
            if (original.start, original.end) not in wanted:
                continue
            if only_original and copy.value(now) != record.value(original):
                continue
            out.setdefault(record.id, set()).add((now.start, now.end))
    return out


def _build(config_path: Path) -> Passport:
    """Run the pipeline and build a passport linked to the previous version."""
    root = config_path.parent.resolve()
    config = load_config(config_path)
    save_run_record(root, run_pipeline(root, config))
    out = root / "passport.json"
    previous = load_previous(out, config.project.name)
    reason = None if previous is None else "Remediated memorized entities and retrained"
    passport = build_passport(config_path, config, previous=previous, reason=reason)
    if previous is not None:
        archive(root, out)
    write_passport(passport, out)
    write_html(passport, out.with_suffix(".html"))
    return passport


def _type_auc(audit: EntityAuditResult) -> dict[str, float]:
    """Attack AUC of the primary method for each entity type."""
    return {
        m.entity_type: m.auc
        for m in audit.metrics
        if m.entity_type and m.method == audit.primary_method
    }


def _round(passport: Passport) -> Round:
    audit = passport.privacy_report.entity_audit if passport.privacy_report else None
    if audit is None:
        raise RemediationError("the passport has no entity audit; set privacy.entity_audit")
    verdict = passport.policy.verdict.value if passport.policy else "none"
    risky = {f.entity_type for f in audit.findings if at_least(f.severity, Severity.HIGH)}
    return Round(
        version=passport.identity.version,
        passport_id=str(passport.identity.passport_id),
        verdict=verdict,
        auc=audit.auc,
        severity_counts=dict(audit.severity_counts),
        risky_types=risky,
        type_auc=_type_auc(audit),
    )


def _sanitize_round(
    root: Path, passport: Passport, version: str, s: RemediationSettings, whole: set[str]
) -> dict[str, int]:
    audit = passport.privacy_report.entity_audit if passport.privacy_report else None
    raw = list(load_corpus(root / RAW_CORPUS))
    train = list(load_corpus(root / TRAIN_CORPUS))
    found = risky_spans(audit, raw, s.min_severity)
    for record_id, spans in risky_spans(None, raw, s.min_severity, (*s.always, *whole)).items():
        found.setdefault(record_id, set()).update(spans)
    targets = translate(found, raw, train, only_original=True)
    history = root / HISTORY_DIR / f"train.v{version}.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / TRAIN_CORPUS, history)
    cleaned, report = sanitize(train, targets, s.strategy, s.seed)
    write_corpus(cleaned, root / TRAIN_CORPUS)
    return dict(report.replaced)


def remediate(
    config_path: Path, settings: RemediationSettings | None = None, echo: Echo | None = None
) -> list[Round]:
    """Build, and while the gate fails, sanitize and rebuild as a new version."""
    s = settings or RemediationSettings()
    say = echo or (lambda _message: None)
    root = config_path.parent.resolve()
    for required in (RAW_CORPUS, TRAIN_CORPUS):
        if not (root / required).is_file():
            raise RemediationError(
                f"{required} not found; create the project with `passport init --llm`"
            )
    rounds: list[Round] = []
    for attempt in range(s.max_rounds + 1):
        passport = _build(config_path)
        current = _round(passport)
        rounds.append(current)
        say(
            f"v{current.version}: verdict {current.verdict}, AUC {current.auc}, "
            f"critical {current.severity_counts.get('critical', 0)}, "
            f"high {current.severity_counts.get('high', 0)}"
        )
        if current.verdict != Verdict.FAIL.value and not current.blocking:
            say(f"release gate passed at v{current.version}")
            break
        if attempt == s.max_rounds:
            say(f"gate still failing after {s.max_rounds} rounds; review the findings")
            break
        previous = rounds[-2].risky_types if len(rounds) > 1 else set()
        current.whole_types = current.risky_types & previous
        if not current.blocking:  # the aggregate attack fails the gate: act on leaky types
            current.whole_types |= {
                kind for kind, auc in current.type_auc.items() if auc > s.type_auc_max
            }
        current.sanitized = _sanitize_round(root, passport, current.version, s, current.whole_types)
        replaced = sum(current.sanitized.values())
        next_version = bump_minor(current.version)
        set_version(config_path, next_version)
        whole = f" (all {', '.join(sorted(current.whole_types))})" if current.whole_types else ""
        say(f"  sanitized {replaced} values{whole}; retraining as v{next_version}")
    return rounds


def gate_passed(rounds: Sequence[Round]) -> bool:
    return bool(rounds) and rounds[-1].verdict != Verdict.FAIL.value and not rounds[-1].blocking
