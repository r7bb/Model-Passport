"""Common scanner interface. Every scanner returns ``Finding`` objects and nothing raw."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pandas as pd

from model_passport.core.schema import Finding

TABULAR_SUFFIXES = {".csv", ".tsv", ".parquet", ".pq", ".jsonl", ".ndjson"}


class ScanError(Exception):
    """Raised when a target cannot be read or scanned."""


def load_table(path: Path) -> pd.DataFrame:
    """Load a CSV, TSV, Parquet, or JSONL file."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, low_memory=False)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", low_memory=False)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix in {".jsonl", ".ndjson"}:
            return pd.read_json(path, lines=True)
    except (OSError, ValueError, ImportError) as exc:
        raise ScanError(f"cannot read {path}: {exc}") from exc
    raise ScanError(f"unsupported table format {suffix!r} for {path}")


@dataclass
class ScanTarget:
    """A file to scan. Tabular data is loaded once and shared across scanners."""

    path: Path
    name: str | None = None
    _frame: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return self.name or self.path.name

    @property
    def is_tabular(self) -> bool:
        return self.path.suffix.lower() in TABULAR_SUFFIXES

    @property
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            self._frame = load_table(self.path)
        return self._frame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, name: str) -> ScanTarget:
        return cls(path=Path(name), name=name, _frame=frame)


class Scanner(ABC):
    """Pluggable scanner. Subclasses set ``name`` and implement ``scan``."""

    name: ClassVar[str]

    @abstractmethod
    def scan(self, target: ScanTarget) -> list[Finding]:
        """Scan one target and return findings (counts, categories, masked examples only)."""
