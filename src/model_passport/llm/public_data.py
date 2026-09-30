"""Public corpora with labeled personal data, converted to MP records.

**Nemotron-PII** (NVIDIA, CC BY 4.0, commercial use allowed with attribution): 100,000 test and
100,000 train documents across industries (medical records, visa forms, bank letters, ...),
each with every personal value labeled by character span. The people in it are synthetic.
``https://huggingface.co/datasets/nvidia/Nemotron-PII``

AI4Privacy is *not* offered here: its license allows academic, non-commercial use only.
"""

from __future__ import annotations

import ast
import json
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from model_passport.llm.entities import Record, Span

NEMOTRON_URL = "https://huggingface.co/datasets/nvidia/Nemotron-PII/resolve/main/data/{split}-00000-of-00001.parquet"
NEMOTRON_ATTRIBUTION = "Nemotron-PII by NVIDIA, licensed under CC BY 4.0"

# Nemotron-PII labels -> MP entity types (the AI4Privacy-style names used for impact and
# look-alike generation). Unlisted labels keep an upper-cased name with the default impact.
NEMOTRON_TYPES = {
    "first_name": "FIRSTNAME",
    "last_name": "LASTNAME",
    "date": "DATE",
    "date_of_birth": "DOB",
    "date_time": "DATE",
    "time": "TIME",
    "company_name": "COMPANYNAME",
    "email": "EMAIL",
    "occupation": "JOBTITLE",
    "url": "URL",
    "phone_number": "PHONENUMBER",
    "fax_number": "PHONENUMBER",
    "state": "STATE",
    "country": "COUNTRY",
    "street_address": "STREET",
    "city": "CITY",
    "county": "COUNTY",
    "postcode": "ZIPCODE",
    "coordinate": "NEARBYGPSCOORDINATE",
    "account_number": "ACCOUNTNUMBER",
    "bank_routing_number": "ACCOUNTNUMBER",
    "user_name": "USERNAME",
    "password": "PASSWORD",
    "pin": "PIN",
    "credit_debit_card": "CREDITCARDNUMBER",
    "cvv": "CREDITCARDCVV",
    "ssn": "SSN",
    "medical_record_number": "MEDICALRECORD",
    "health_plan_beneficiary_number": "MEDICALRECORD",
    "swift_bic": "BIC",
    "ipv4": "IPV4",
    "ipv6": "IPV6",
    "mac_address": "MAC",
    "license_plate": "VEHICLEVRM",
    "vehicle_identifier": "VEHICLEVIN",
    "age": "AGE",
    "gender": "GENDER",
}


def download_nemotron(directory: Path, split: str = "test") -> Path:
    """Download one split (about 150 MB) once, and return its local path."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"nemotron-pii-{split}.parquet"
    if not target.exists():
        partial = target.with_suffix(".part")
        urllib.request.urlretrieve(NEMOTRON_URL.format(split=split), partial)  # noqa: S310 - fixed https URL
        partial.rename(target)
    return target


def _spans(raw: str, text: str) -> list[Span]:
    spans = []
    for item in ast.literal_eval(raw) if raw.startswith("[{'") else json.loads(raw):
        start, end = int(item["start"]), int(item["end"])
        if 0 <= start < end <= len(text) and text[start:end].strip():
            label = str(item["label"])
            spans.append(Span(start, end, NEMOTRON_TYPES.get(label, label.upper())))
    # Keep the earliest of overlapping spans, so every character has at most one label.
    kept: list[Span] = []
    for span in sorted(spans, key=lambda s: (s.start, -s.end)):
        if not kept or span.start >= kept[-1].end:
            kept.append(span)
    return kept


def nemotron_records(
    path: Path, limit: int | None = None, max_chars: int = 1_000, locale: str = "us"
) -> Iterator[Record]:
    """Documents of one locale up to ``max_chars`` long, with their labeled values."""
    table = pq.read_table(path, columns=["uid", "locale", "text", "spans"])
    produced = 0
    for batch in table.to_batches(max_chunksize=4_096):
        for row in batch.to_pylist():
            text = str(row["text"])
            if row["locale"] != locale or len(text) > max_chars:
                continue
            spans = _spans(str(row["spans"]), text)
            if spans:
                yield Record(str(row["uid"]), text, spans)
                produced += 1
                if limit is not None and produced >= limit:
                    return
