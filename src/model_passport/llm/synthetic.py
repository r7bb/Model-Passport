"""Synthetic training corpora with annotated sensitive entities, for demos and tests.

Every value comes from Faker: nothing describes a real person. Records look like support
tickets, billing notes, and HR messages, with one to three entities each.
"""

from __future__ import annotations

import random

from faker import Faker

from model_passport.llm.entities import Record, Span

TEMPLATES = [
    "Customer {FULLNAME} called about a late refund; call back on {PHONENUMBER}.",
    "Please send the invoice to {EMAIL} before Friday.",
    "The account holder {FULLNAME} asked to update the billing address to {STREET}.",
    "Our records show {FULLNAME} renewed the plan in March using card {CREDITCARDNUMBER}.",
    "Verified identity for ticket 4471 with SSN {SSN} and date of birth {DOB}.",
    "New hire {FULLNAME} starts Monday; payroll email is {EMAIL}.",
    "Login alerts for {USERNAME} came from IP address {IPV4}.",
    "Escalated: {FULLNAME} ({EMAIL}) reports duplicate charges.",
    "Ship the replacement to {STREET}, {CITY}, and text {PHONENUMBER} on delivery.",
    "Wire refund to IBAN {IBAN} for account holder {FULLNAME}.",
]


def _generators(fake: Faker) -> dict[str, object]:
    return {
        "FULLNAME": fake.name,
        "PHONENUMBER": fake.phone_number,
        "EMAIL": fake.email,
        "STREET": fake.street_address,
        "CREDITCARDNUMBER": fake.credit_card_number,
        "SSN": fake.ssn,
        "DOB": lambda: fake.date_of_birth().isoformat(),
        "USERNAME": fake.user_name,
        "IPV4": fake.ipv4,
        "CITY": fake.city,
        "IBAN": fake.iban,
    }


def fill(template: str, values: dict[str, str]) -> Record:
    """Fill a ``{TYPE}`` template and record each value's span."""
    text, spans = "", []
    for part in _parse(template):
        if part.startswith("{") and part.endswith("}"):
            kind = part[1:-1]
            value = values[kind]
            spans.append(Span(len(text), len(text) + len(value), kind))
            text += value
        else:
            text += part
    return Record("", text, spans)


def _parse(template: str) -> list[str]:
    parts, buffer, index = [], "", 0
    while index < len(template):
        if template[index] == "{":
            end = template.index("}", index)
            parts.extend([buffer, template[index : end + 1]])
            buffer, index = "", end + 1
        else:
            buffer += template[index]
            index += 1
    parts.append(buffer)
    return [p for p in parts if p]


def corpus(n: int, seed: int = 0) -> list[Record]:
    """``n`` records with fresh synthetic values."""
    rng = random.Random(seed)  # noqa: S311 - reproducible synthetic data
    fake = Faker()
    fake.seed_instance(seed)
    generate = _generators(fake)
    records = []
    for i in range(n):
        template = rng.choice(TEMPLATES)
        values = {kind: str(fn()) for kind, fn in generate.items()}  # type: ignore[operator]
        record = fill(template, values)
        records.append(Record(f"r{i}", record.text, record.entities))
    return records
