"""SQLite storage for the passport registry."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS passports (
    passport_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    verdict TEXT,
    merkle_root TEXT NOT NULL,
    key_fingerprint TEXT NOT NULL,
    supersedes TEXT,
    public_key_pem TEXT NOT NULL,
    document TEXT NOT NULL,
    uploaded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_passports_model ON passports (model_name, created_at);
"""


@dataclass(frozen=True)
class StoredPassport:
    passport_id: str
    model_name: str
    version: str
    created_at: str
    verdict: str | None
    merkle_root: str
    key_fingerprint: str
    supersedes: str | None
    public_key_pem: str
    document: dict[str, Any]
    uploaded_at: str

    def summary(self) -> dict[str, Any]:
        events = self.document.get("events") or []
        return {
            "passport_id": self.passport_id,
            "model_name": self.model_name,
            "version": self.version,
            "created_at": self.created_at,
            "verdict": self.verdict,
            "merkle_root": self.merkle_root,
            "key_fingerprint": self.key_fingerprint,
            "supersedes": self.supersedes,
            "events": len(events),
            "uploaded_at": self.uploaded_at,
        }


class DuplicatePassportError(Exception):
    """Raised when a passport ID is uploaded twice."""


class RegistryDB:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # One shared connection; FastAPI runs sync endpoints in a thread pool.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._conn:
            yield self._conn

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> StoredPassport:
        data = dict(row)
        data["document"] = json.loads(data["document"])
        return StoredPassport(**data)

    def insert(self, document: dict[str, Any], public_key_pem: str) -> StoredPassport:
        ident = document["identity"]
        policy = document.get("policy") or {}
        revision = document.get("revision") or {}
        try:
            with self._tx() as conn:
                conn.execute(
                    "INSERT INTO passports (passport_id, model_name, version, created_at, verdict,"
                    " merkle_root, key_fingerprint, supersedes, public_key_pem, document)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ident["passport_id"], ident["model_name"], ident["version"],
                        ident["created_at"], policy.get("verdict"), ident["merkle_root"],
                        ident["public_key_fingerprint"], revision.get("previous_passport_id"),
                        public_key_pem, json.dumps(document),
                    ),
                )  # fmt: skip
        except sqlite3.IntegrityError as exc:
            raise DuplicatePassportError(ident["passport_id"]) from exc
        stored = self.get(ident["passport_id"])
        if stored is None:  # pragma: no cover - the row was just inserted
            raise RuntimeError(f"passport {ident['passport_id']} vanished after insert")
        return stored

    def replace_document(self, passport_id: str, document: dict[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE passports SET document = ? WHERE passport_id = ?",
                (json.dumps(document), passport_id),
            )

    def get(self, passport_id: str) -> StoredPassport | None:
        row = self._conn.execute(
            "SELECT * FROM passports WHERE passport_id = ?", (passport_id,)
        ).fetchone()
        return self._row(row) if row else None

    def search(
        self, model_name: str | None = None, verdict: str | None = None, limit: int = 200
    ) -> list[StoredPassport]:
        clauses, params = [], []
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        if verdict:
            clauses.append("verdict = ?")
            params.append(verdict)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM passports {where} ORDER BY created_at DESC LIMIT ?",  # noqa: S608
            (*params, limit),
        ).fetchall()
        return [self._row(r) for r in rows]

    def models(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT model_name, COUNT(*) AS passports, MAX(created_at) AS latest"
            " FROM passports GROUP BY model_name ORDER BY model_name"
        ).fetchall()
        return [dict(r) for r in rows]
