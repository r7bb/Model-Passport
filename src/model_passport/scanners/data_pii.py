"""Tabular PII scanner: column name heuristics plus value-level validators.

Findings carry counts, rates, and masked examples only. Raw values never leave this module.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from model_passport.core.schema import Finding, Severity
from model_passport.scanners.base import Scanner, ScanTarget

# --- Entity catalogue ----------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    name: str
    severity: Severity
    direct_identifier: bool


ENTITIES = {
    e.name: e
    for e in [
        Entity("EMAIL", Severity.HIGH, True),
        Entity("PHONE", Severity.HIGH, True),
        Entity("US_SSN", Severity.CRITICAL, True),
        Entity("CREDIT_CARD", Severity.CRITICAL, True),
        Entity("PERSON_NAME", Severity.HIGH, True),
        Entity("ADDRESS", Severity.HIGH, True),
        Entity("MEDICAL_RECORD_NUMBER", Severity.HIGH, True),
        Entity("DATE_OF_BIRTH", Severity.HIGH, True),
        Entity("IP_ADDRESS", Severity.MEDIUM, False),
        Entity("ZIP_CODE", Severity.LOW, False),
        Entity("DATE", Severity.LOW, False),
    ]
}

# Column name tokens -> entity. Names are split on non-alphanumerics and camelCase.
NAME_HINTS: list[tuple[set[str], str]] = [
    ({"email", "e_mail", "mail"}, "EMAIL"),
    ({"phone", "mobile", "cell", "tel", "telephone", "fax"}, "PHONE"),
    ({"ssn", "social_security", "socialsecurity"}, "US_SSN"),
    ({"credit_card", "creditcard", "card_number", "ccn", "pan"}, "CREDIT_CARD"),
    ({"name", "firstname", "lastname", "fullname", "surname", "first_name", "last_name",
      "full_name", "patient_name"}, "PERSON_NAME"),
    ({"address", "street", "addr", "address_line"}, "ADDRESS"),
    ({"mrn", "medical_record", "medical_record_number"}, "MEDICAL_RECORD_NUMBER"),
    ({"dob", "birth", "birthdate", "birthday", "date_of_birth", "birth_date"}, "DATE_OF_BIRTH"),
    ({"ip", "ip_address", "ipaddress", "ipv4", "ipv6"}, "IP_ADDRESS"),
    ({"zip", "zipcode", "zip_code", "postal", "postcode", "postal_code"}, "ZIP_CODE"),
]  # fmt: skip


def column_tokens(column: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", column).lower()
    parts = [p for p in re.split(r"[^a-z0-9]+", spaced) if p]
    tokens = set(parts)
    tokens.update("_".join(parts[i : i + 2]) for i in range(len(parts) - 1))
    tokens.add("_".join(parts))
    return tokens


def entity_from_column_name(column: str) -> str | None:
    tokens = column_tokens(column)
    for hints, entity in NAME_HINTS:
        if tokens & hints:
            return entity
    return None


# --- Value validators ----------------------------------------------------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"(?<![\w-])(?:(?:\+1|001)[-.\s]?)?(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}"
    r"(?:\s?(?:x|ext\.?)\s?\d{1,6})?(?![\w-])"
)
BARE_PHONE_RE = re.compile(r"^\s*\+?1?\d{10}\s*$")
SSN_RE = re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d-])")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6_RE = re.compile(r"(?<![:\w])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![:\w])")
DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})\b")
CARD_MIN_HIT_RATE = 0.5


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _find_cards(text: str) -> list[str]:
    hits = []
    for match in CARD_RE.finditer(text):
        if PHONE_RE.fullmatch(match.group().strip()):
            continue
        digits = re.sub(r"[ -]", "", match.group())
        if 13 <= len(digits) <= 19 and luhn_valid(digits) and len(set(digits)) > 1:
            hits.append(match.group())
    return hits


def _find_ips(text: str) -> list[str]:
    hits = []
    for match in IPV4_RE.finditer(text):
        try:
            ipaddress.IPv4Address(match.group())
            hits.append(match.group())
        except ValueError:
            pass
    for match in IPV6_RE.finditer(text):
        if match.group().count(":") < 2:
            continue
        try:
            ipaddress.IPv6Address(match.group())
            hits.append(match.group())
        except ValueError:
            pass
    return hits


def _parse_date(text: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _find_dates(text: str) -> list[str]:
    today = date.today()
    hits = []
    for match in DATE_RE.finditer(text):
        parsed = _parse_date(match.group())
        if parsed and date(today.year - 120, 1, 1) <= parsed <= today:
            hits.append(match.group())
    return hits


VALIDATORS: dict[str, Callable[[str], list[str]]] = {
    "EMAIL": EMAIL_RE.findall,
    "US_SSN": SSN_RE.findall,
    "CREDIT_CARD": _find_cards,
    "PHONE": PHONE_RE.findall,
    "IP_ADDRESS": _find_ips,
    "DATE": _find_dates,
}


# --- Masking -------------------------------------------------------------------------------


def mask_value(value: str, entity: str) -> str:
    """Mask a detected value, e.g. ``jane@gmail.com`` -> ``j***@g***.com``."""
    if entity == "EMAIL" and "@" in value:
        local, _, domain = value.partition("@")
        host, _, tld = domain.rpartition(".")
        return f"{local[:1]}***@{host[:1]}***.{tld}"
    if entity in {"PHONE", "US_SSN", "CREDIT_CARD"}:
        keep = 2
        digits_seen = sum(ch.isdigit() for ch in value)
        out, seen = [], 0
        for ch in value:
            if ch.isdigit():
                seen += 1
                out.append(ch if seen > digits_seen - keep else "*")
            else:
                out.append(ch)
        return "".join(out)
    return re.sub(r"[A-Za-z0-9]", "*", value[:1]) + re.sub(r"[A-Za-z0-9]", "*", value[1:])


# --- Scanner -------------------------------------------------------------------------------


class PiiScanner(Scanner):
    """Detect direct identifiers and PII-bearing columns in a table."""

    name = "pii"

    def __init__(
        self,
        sample_size: int | None = 10_000,
        seed: int = 0,
        max_examples: int = 2,
        use_presidio: bool = False,
    ) -> None:
        self.sample_size = sample_size
        self.seed = seed
        self.max_examples = max_examples
        self.use_presidio = use_presidio

    def _sample(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.sample_size is None or len(frame) <= self.sample_size:
            return frame
        return frame.sample(n=self.sample_size, random_state=self.seed)

    def scan(self, target: ScanTarget) -> list[Finding]:
        frame = self._sample(target.frame)
        findings: list[Finding] = []
        for column in frame.columns:
            findings.extend(self._scan_column(target.label, str(column), frame[column]))
        return findings

    def _scan_column(self, dataset: str, column: str, series: pd.Series) -> list[Finding]:
        values = series.dropna().astype(str)
        values = values[values.str.strip() != ""]
        name_entity = entity_from_column_name(column)
        hits = self._collect_hits(values, name_entity) if len(values) else {}
        _apply_context_rules(hits, name_entity, values)

        findings = [
            self._finding(
                dataset, column, entity, count, count / len(values), entity == name_entity, examples
            )
            for entity, (count, examples) in hits.items()
        ]
        if name_entity and name_entity not in hits:
            # Name-only evidence (e.g. a "name" column, which regex cannot validate).
            findings.append(
                self._finding(dataset, column, name_entity, len(values), None, True, [])
            )
        return findings

    def _collect_hits(self, values: pd.Series, name_entity: str | None) -> _Hits:
        """Rows matching each validator, with a few masked examples."""
        validators = dict(VALIDATORS)
        if name_entity == "PHONE":
            # Bare digit strings are only phone numbers when the column says so.
            validators["PHONE"] = lambda s: PHONE_RE.findall(s) or BARE_PHONE_RE.findall(s)
        hits: _Hits = {}
        for entity, find in validators.items():
            count, examples = 0, list[str]()
            for value in values:
                matches = find(value)
                if matches:
                    count += 1
                    if len(examples) < self.max_examples:
                        examples.append(mask_value(matches[0], entity))
            if count:
                hits[entity] = (count, examples)
        if self.use_presidio and _is_free_text(values):
            for entity, count in _presidio_counts(values).items():
                hits.setdefault(entity, (count, []))
        return hits

    def _finding(
        self,
        dataset: str,
        column: str,
        entity_name: str,
        count: int,
        hit_rate: float | None,
        name_agrees: bool,
        examples: list[str],
    ) -> Finding:
        entity = ENTITIES.get(entity_name, Entity(entity_name, Severity.MEDIUM, False))
        if hit_rate is None:
            confidence = 0.6
            evidence = "column name"
        else:
            confidence = max(hit_rate, 0.9) if name_agrees else hit_rate
            evidence = "column name and values" if name_agrees else "values"
        return Finding(
            scanner=self.name,
            category=entity.name,
            severity=entity.severity,
            location=f"{dataset}:{column}",
            count=count,
            message=f"{entity.name} detected in column {column!r} ({evidence})",
            masked_examples=examples,
            details={
                "dataset": dataset,
                "column": column,
                "hit_rate": None if hit_rate is None else round(hit_rate, 4),
                "confidence": round(confidence, 4),
                "direct_identifier": entity.direct_identifier,
                "evidence": evidence,
            },
        )


_Hits = dict[str, tuple[int, list[str]]]


def _apply_context_rules(hits: _Hits, name_entity: str | None, values: pd.Series) -> None:
    """Reinterpret or drop hits using the column name and content type."""
    # Dates in a birth-named column are dates of birth.
    if "DATE" in hits and name_entity == "DATE_OF_BIRTH":
        hits["DATE_OF_BIRTH"] = hits.pop("DATE")
    # Random digit strings pass the Luhn check 10% of the time, so a column of numeric IDs
    # would look like sparse card numbers. Require a majority unless the name or free-text
    # context supports it.
    if (
        "CREDIT_CARD" in hits
        and name_entity != "CREDIT_CARD"
        and hits["CREDIT_CARD"][0] / len(values) < CARD_MIN_HIT_RATE
        and not _is_free_text(values)
    ):
        del hits["CREDIT_CARD"]


def _is_free_text(values: pd.Series) -> bool:
    sample = values.head(200)
    return bool(len(sample)) and float(sample.str.contains(r"\s").mean()) > 0.5


def _presidio_counts(values: pd.Series) -> dict[str, int]:
    """Entity counts from Microsoft Presidio, if installed. Returns {} otherwise."""
    try:
        from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415 - optional dependency
    except ImportError:
        return {}
    analyzer = AnalyzerEngine()
    counts: dict[str, int] = {}
    for value in values.head(2000):
        for entity in {r.entity_type for r in analyzer.analyze(text=value, language="en")}:
            counts[entity] = counts.get(entity, 0) + 1
    return counts


def pii_columns(findings: list[Finding]) -> set[str]:
    """Columns (``dataset:column``) holding direct identifiers."""
    return {
        f.location or ""
        for f in findings
        if f.scanner == PiiScanner.name and f.details.get("direct_identifier")
    }
