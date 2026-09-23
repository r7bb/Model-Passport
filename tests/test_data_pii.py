from __future__ import annotations

import json

import pandas as pd
import pytest
from faker import Faker

from model_passport.scanners.base import ScanError, ScanTarget, load_table
from model_passport.scanners.data_pii import (
    PiiScanner,
    column_tokens,
    entity_from_column_name,
    luhn_valid,
    mask_value,
    pii_columns,
)

fake = Faker("en_US")
Faker.seed(1)


def _scan(frame: pd.DataFrame, **kwargs: object) -> dict[str, dict[str, object]]:
    findings = PiiScanner(**kwargs).scan(ScanTarget.from_frame(frame, "t"))
    return {f"{f.details['column']}:{f.category}": f for f in findings}


def test_column_name_heuristics() -> None:
    assert entity_from_column_name("Email") == "EMAIL"
    assert entity_from_column_name("customerPhone") == "PHONE"
    assert entity_from_column_name("patient_name") == "PERSON_NAME"
    assert entity_from_column_name("DOB") == "DATE_OF_BIRTH"
    assert entity_from_column_name("zip_code") == "ZIP_CODE"
    assert entity_from_column_name("mrn") == "MEDICAL_RECORD_NUMBER"
    assert entity_from_column_name("hours_per_week") is None
    assert entity_from_column_name("usage") is None  # "age" substring must not match
    assert "first_name" in column_tokens("firstName")


def test_luhn() -> None:
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")


def test_masking_never_returns_raw_value() -> None:
    assert mask_value("jane.doe@gmail.com", "EMAIL") == "j***@g***.com"
    assert mask_value("123-45-6789", "US_SSN") == "***-**-**89"
    assert mask_value("4111 1111 1111 1111", "CREDIT_CARD") == "**** **** **** **11"
    assert mask_value("2001-02-03", "DATE_OF_BIRTH") == "****-**-**"


def test_detects_injected_pii_with_faker() -> None:
    n = 200
    frame = pd.DataFrame(
        {
            "name": [fake.name() for _ in range(n)],
            "contact": [fake.email() for _ in range(n)],
            "phone": [fake.phone_number() for _ in range(n)],
            "ssn": [fake.ssn() for _ in range(n)],
            "cc": [fake.credit_card_number(card_type="visa16") for _ in range(n)],
            "dob": [fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat()
                    for _ in range(n)],
            "ip": [fake.ipv4() for _ in range(n)],
            "age": [fake.random_int(18, 90) for _ in range(n)],
            "note": ["call me at " + fake.email() if i % 10 == 0 else "fine" for i in range(n)],
        }
    )  # fmt: skip
    hits = _scan(frame)
    assert hits["name:PERSON_NAME"].details["evidence"] == "column name"
    assert hits["contact:EMAIL"].count == n
    assert hits["phone:PHONE"].count == n
    assert hits["ssn:US_SSN"].count == n
    assert hits["cc:CREDIT_CARD"].count == n
    assert hits["dob:DATE_OF_BIRTH"].count == n
    assert hits["ip:IP_ADDRESS"].count == n
    assert hits["note:EMAIL"].details["hit_rate"] == pytest.approx(0.1)
    assert not [k for k in hits if k.startswith("age:")]

    for finding in hits.values():
        blob = json.dumps(finding.model_dump(mode="json"))
        for column in ("name", "contact", "ssn", "cc", "phone"):
            for raw in frame[column].head(50):
                assert str(raw) not in blob, f"raw value leaked from {column}"


def test_clean_frame_has_no_direct_identifiers() -> None:
    frame = pd.DataFrame(
        {
            "age": ["30-39"] * 50,
            "zip": ["021**"] * 50,
            "hours_per_week": list(range(50)),
            "account_ms": [1_700_000_000_000 + i * 7919 for i in range(50)],  # 13-digit numbers
        }
    )
    findings = PiiScanner().scan(ScanTarget.from_frame(frame, "clean"))
    assert pii_columns(findings) == set()
    assert {f.category for f in findings} == {"ZIP_CODE"}


def test_sampling_limits_rows() -> None:
    frame = pd.DataFrame({"email": [fake.email() for _ in range(500)]})
    (finding,) = PiiScanner(sample_size=100).scan(ScanTarget.from_frame(frame, "t"))
    assert finding.count == 100


def test_load_table_formats(tmp_path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    frame.to_csv(tmp_path / "t.csv", index=False)
    frame.to_json(tmp_path / "t.jsonl", orient="records", lines=True)
    frame.to_parquet(tmp_path / "t.parquet")
    for name in ("t.csv", "t.jsonl", "t.parquet"):
        assert load_table(tmp_path / name).shape == (2, 2)
    (tmp_path / "t.xlsx").write_text("nope")
    with pytest.raises(ScanError):
        load_table(tmp_path / "t.xlsx")
