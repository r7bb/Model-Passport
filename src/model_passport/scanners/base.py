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


STRING_DTYPES = {"object", "str", "string"}


def string_columns(schema: dict[str, str] | None) -> list[str]:
    """Columns a model schema declares as strings (e.g. ZIP codes with leading zeros)."""
    return [col for col, dtype in (schema or {}).items() if str(dtype).lower() in STRING_DTYPES]


def load_table(path: Path, string_cols: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV, TSV, Parquet, or JSONL file.

    ``string_cols`` are read as text so type inference cannot corrupt them (``"02103"`` must not
    become ``2103``). Pass the columns a model was trained on as strings.
    """
    suffix = path.suffix.lower()
    as_text = dict.fromkeys(string_cols or [], str)
    try:
        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            header = pd.read_csv(path, sep=sep, nrows=0).columns
            dtype = {c: t for c, t in as_text.items() if c in header}
            return pd.read_csv(path, sep=sep, low_memory=False, dtype=dtype or None)
        if suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        elif suffix in {".jsonl", ".ndjson"}:
            frame = pd.read_json(path, lines=True, dtype=False)
        else:
            raise ScanError(f"unsupported table format {suffix!r} for {path}")
    except (OSError, ValueError, ImportError) as exc:
        raise ScanError(f"cannot read {path}: {exc}") from exc
    for column in as_text:
        if column in frame.columns:
            frame[column] = frame[column].where(frame[column].isna(), frame[column].astype(str))
    return frame


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
