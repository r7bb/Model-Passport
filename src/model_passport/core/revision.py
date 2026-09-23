"""Link a rebuilt passport (new data, retraining) to the version it supersedes."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from model_passport.core.schema import ChangeStatus, DatasetChange, Passport, RevisionInfo

HISTORY_DIR = Path(".passport/history")


def load_previous(path: Path, model_name: str) -> Passport | None:
    """The existing passport at ``path`` if it describes the same model, else None."""
    if not path.is_file():
        return None
    try:
        previous = Passport.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError):
        return None
    return previous if previous.identity.model_name == model_name else None


def archive(root: Path, passport_path: Path) -> Path:
    """Copy a superseded passport (with its events) into ``.passport/history``."""
    data = json.loads(passport_path.read_text(encoding="utf-8"))
    target = root / HISTORY_DIR / f"{data['identity']['passport_id']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(passport_path.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _dataset_changes(previous: Passport, current: Passport) -> list[DatasetChange]:
    before = {d.name: d for d in previous.datasets}
    after = {d.name: d for d in current.datasets}
    changes = []
    for name in sorted(before.keys() | after.keys()):
        old, new = before.get(name), after.get(name)
        if old is None:
            status = ChangeStatus.ADDED
        elif new is None:
            status = ChangeStatus.REMOVED
        elif old.sha256 != new.sha256:
            status = ChangeStatus.CHANGED
        else:
            status = ChangeStatus.UNCHANGED
        changes.append(
            DatasetChange(
                name=name,
                status=status,
                previous_sha256=old.sha256 if old else None,
                current_sha256=new.sha256 if new else None,
                previous_rows=old.row_count if old else None,
                current_rows=new.row_count if new else None,
            )
        )
    return changes


def _metric_deltas(previous: Passport, current: Passport) -> dict[str, dict[str, float]]:
    deltas: dict[str, dict[str, float]] = {}
    for split, values in current.metrics.items():
        for name, value in values.items():
            old = previous.metrics.get(split, {}).get(name)
            if old is not None:
                deltas.setdefault(split, {})[name] = round(value - old, 6)
    return deltas


def compute_revision(previous: Passport, current: Passport, reason: str | None) -> RevisionInfo:
    old = {a.path: a.sha256 for a in previous.artifacts}
    new = {a.path: a.sha256 for a in current.artifacts}
    return RevisionInfo(
        previous_passport_id=previous.identity.passport_id,
        previous_version=previous.identity.version,
        previous_merkle_root=previous.identity.merkle_root,
        previous_created_at=previous.identity.created_at,
        sequence=(previous.revision.sequence + 1) if previous.revision else 1,
        reason=reason,
        added_artifacts=sorted(new.keys() - old.keys()),
        removed_artifacts=sorted(old.keys() - new.keys()),
        changed_artifacts=sorted(p for p in new.keys() & old.keys() if new[p] != old[p]),
        dataset_changes=_dataset_changes(previous, current),
        metric_deltas=_metric_deltas(previous, current),
    )
