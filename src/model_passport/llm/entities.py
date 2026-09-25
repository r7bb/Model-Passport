"""Sensitive entities in training text: types, harm levels, detection, and corpus loading.

Entity types follow the AI4Privacy labels used by the EL-MIA benchmark (``EMAIL``,
``PHONENUMBER``, ``SSN``, ...), so annotated corpora in that format audit without mapping.

Impact is the harm if a value of that type leaks, following the confidentiality impact levels
of NIST SP 800-122 (Guide to Protecting the Confidentiality of PII): 1.0 for values that enable
fraud or account takeover on their own, 0.8 for direct identifiers, 0.6 for values that
identify in combination, and 0.4 for low-sensitivity attributes.
"""

from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from model_passport.scanners.data_pii import (
    CARD_RE,
    DATE_RE,
    EMAIL_RE,
    IPV4_RE,
    IPV6_RE,
    PHONE_RE,
    SSN_RE,
    luhn_valid,
    mask_value,
)

HIGH, MODERATE_HIGH, MODERATE, LOW = 1.0, 0.8, 0.6, 0.4
IMPACT: dict[str, float] = {
    **dict.fromkeys(
        ["SSN", "CREDITCARDNUMBER", "CREDITCARDCVV", "PASSWORD", "PIN", "IBAN", "BIC",
         "ACCOUNTNUMBER", "MASKEDNUMBER", "BITCOINADDRESS", "ETHEREUMADDRESS", "LITECOINADDRESS",
         "PHONEIMEI", "VEHICLEVIN", "MEDICALRECORD"],
        HIGH,
    ),
    **dict.fromkeys(
        ["EMAIL", "PHONENUMBER", "DOB", "FULLNAME", "STREET", "BUILDINGNUMBER",
         "SECONDARYADDRESS", "USERNAME", "IPV4", "IPV6", "IP", "MAC", "NEARBYGPSCOORDINATE",
         "VEHICLEVRM", "ACCOUNTNAME"],
        MODERATE_HIGH,
    ),
    **dict.fromkeys(
        ["FIRSTNAME", "LASTNAME", "MIDDLENAME", "ZIPCODE", "CITY", "DATE", "AGE", "URL",
         "JOBTITLE", "JOBAREA", "JOBTYPE", "COMPANYNAME", "USERAGENT", "AMOUNT", "TIME"],
        MODERATE,
    ),
}  # fmt: skip
DEFAULT_IMPACT = LOW  # SEX, GENDER, STATE, COUNTY, EYECOLOR, HEIGHT, CURRENCY, ...

# Labels used by the value masker for each entity type.
_MASK_KIND = {
    "EMAIL": "EMAIL", "PHONENUMBER": "PHONE", "SSN": "US_SSN",
    "CREDITCARDNUMBER": "CREDIT_CARD", "DOB": "DATE_OF_BIRTH", "DATE": "DATE",
}  # fmt: skip


def impact(entity_type: str) -> float:
    return IMPACT.get(entity_type.upper(), DEFAULT_IMPACT)


def mask(value: str, entity_type: str) -> str:
    """A masked form safe for reports, e.g. ``j***@e***.com`` or ``J*** D***``."""
    return mask_value(value, _MASK_KIND.get(entity_type.upper(), entity_type.upper()))


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    entity_type: str


@dataclass
class Record:
    """One training text and the sensitive entities in it."""

    id: str
    text: str
    entities: list[Span] = field(default_factory=list)

    def value(self, span: Span) -> str:
        return self.text[span.start : span.end]


# --- Detection -------------------------------------------------------------------------------


def _valid_ip(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
    except ValueError:
        return False
    return True


def _valid_card(text: str) -> bool:
    digits = re.sub(r"[ -]", "", text)
    return 13 <= len(digits) <= 19 and luhn_valid(digits) and len(set(digits)) > 1


_DETECTORS: list[tuple[str, re.Pattern[str], object]] = [
    ("EMAIL", EMAIL_RE, None),
    ("SSN", SSN_RE, None),
    ("CREDITCARDNUMBER", CARD_RE, _valid_card),
    ("PHONENUMBER", PHONE_RE, None),
    ("IPV4", IPV4_RE, _valid_ip),
    ("IPV6", IPV6_RE, _valid_ip),
    ("DATE", DATE_RE, None),
]


def detect(text: str) -> list[Span]:
    """Pattern-based sensitive spans (emails, SSNs, cards, phones, IPs, dates), non-overlapping.

    Names, addresses, and other free-text entities cannot be found reliably by patterns; supply
    annotated corpora (AI4Privacy format) or pre-annotated spans for those.
    """
    found: list[Span] = []
    for entity_type, pattern, validate in _DETECTORS:
        for match in pattern.finditer(text):
            value = match.group().strip()
            if not value or (callable(validate) and not validate(value)):
                continue
            start = match.start() + (len(match.group()) - len(match.group().lstrip()))
            found.append(Span(start, start + len(value), entity_type))
    # Earlier detectors win on overlap (an SSN or card is also digit-shaped like a phone).
    kept: list[Span] = []
    for span in found:
        if all(span.end <= k.start or span.start >= k.end for k in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)


# --- Corpus loading ----------------------------------------------------------------------------


def _spans(raw: Iterable[dict[str, object]]) -> list[Span]:
    spans = []
    for item in raw:
        entity_type = str(item.get("type") or item.get("label") or "UNKNOWN").upper()
        start, end = item.get("start"), item.get("end")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            spans.append(Span(start, end, entity_type))
    return sorted(spans, key=lambda s: s.start)


def record_from_json(data: dict[str, object], index: int, detect_missing: bool = True) -> Record:
    """A record from our format (``text``, ``entities``) or AI4Privacy (``source_text``,
    ``privacy_mask``). Records without annotations are detected with patterns."""
    text = str(data.get("text", data.get("source_text", "")))
    raw = data.get("entities", data.get("privacy_mask"))
    spans = _spans(raw) if isinstance(raw, list) else (detect(text) if detect_missing else [])
    record_id = str(data.get("id", data.get("uid", index)))
    return Record(record_id, text, [s for s in spans if s.end <= len(text)])


def load_corpus(path: Path, detect_missing: bool = True) -> Iterator[Record]:
    """Records from JSONL (one object per line) or plain text (one record per line)."""
    with path.open(encoding="utf-8") as fh:
        for index, raw in enumerate(fh):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if path.suffix.lower() in {".jsonl", ".ndjson", ".json"}:
                yield record_from_json(json.loads(line), index, detect_missing)
            else:
                yield Record(str(index), line, detect(line) if detect_missing else [])


def write_corpus(records: Iterable[Record], path: Path) -> None:
    """Write records as JSONL in our format (used for sanitized corpora)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            entities = [
                {"start": s.start, "end": s.end, "type": s.entity_type} for s in record.entities
            ]
            fh.write(json.dumps({"id": record.id, "text": record.text, "entities": entities}))
            fh.write("\n")
