"""Plausible alternatives for a sensitive value: the reference sets of the EL-MIA attacks.

A reference must be the same type as the candidate and look like it (similar length, casing,
digit layout), so the model has no surface reason to prefer the real value. Values come from
Faker and format grammars (Luhn-valid cards, valid SSN ranges, E.164-style phones), or from
other values of the same type in the corpus.

``source`` mirrors the EL-MIA benchmark subsets: ``synthetic`` alternatives were never trained
on (the "untrained" subset, easiest for the attacker), ``corpus`` alternatives were trained on
elsewhere (the "trained" subset, hardest), and ``mix`` draws from both at equal rates.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable

from faker import Faker

from model_passport.llm.entities import Record

MAX_DRAWS = 40  # attempts per reference to find a surface-matched value
LENGTH_TOLERANCE = 0.25


def _card(fake: Faker) -> str:
    return str(fake.credit_card_number())


def _generators(fake: Faker) -> dict[str, Callable[[], str]]:
    return {
        "EMAIL": fake.email,
        "PHONENUMBER": fake.phone_number,
        "SSN": fake.ssn,
        "CREDITCARDNUMBER": lambda: _card(fake),
        "CREDITCARDCVV": fake.credit_card_security_code,
        "IPV4": fake.ipv4,
        "IPV6": fake.ipv6,
        "IP": fake.ipv4,
        "MAC": fake.mac_address,
        "DATE": fake.date,
        "DOB": lambda: fake.date_of_birth().isoformat(),
        "TIME": fake.time,
        "FIRSTNAME": fake.first_name,
        "MIDDLENAME": fake.first_name,
        "LASTNAME": fake.last_name,
        "FULLNAME": fake.name,
        "PREFIX": fake.prefix,
        "USERNAME": fake.user_name,
        "PASSWORD": fake.password,
        "PIN": lambda: fake.numerify("####"),
        "URL": fake.url,
        "STREET": fake.street_name,
        "BUILDINGNUMBER": fake.building_number,
        "SECONDARYADDRESS": fake.secondary_address,
        "CITY": fake.city,
        "STATE": fake.state,
        "COUNTY": lambda: f"{fake.last_name()} County",
        "ZIPCODE": fake.postcode,
        "NEARBYGPSCOORDINATE": lambda: f"[{fake.latitude()}, {fake.longitude()}]",
        "IBAN": fake.iban,
        "BIC": fake.swift,
        "ACCOUNTNUMBER": fake.bban,
        "ACCOUNTNAME": lambda: f"{fake.word().title()} Account",
        "AMOUNT": lambda: f"{fake.pyfloat(left_digits=4, right_digits=2, positive=True):.2f}",
        "CURRENCYCODE": fake.currency_code,
        "CURRENCYNAME": fake.currency_name,
        "CURRENCYSYMBOL": fake.currency_symbol,
        "COMPANYNAME": fake.company,
        "JOBTITLE": fake.job,
        "JOBAREA": lambda: fake.job().split()[-1],
        "JOBTYPE": lambda: fake.random_element(["Manager", "Engineer", "Analyst", "Director"]),
        "AGE": lambda: str(fake.random_int(18, 90)),
        "SEX": lambda: fake.random_element(["male", "female"]),
        "GENDER": lambda: fake.random_element(["Male", "Female", "Non-binary"]),
        "EYECOLOR": lambda: fake.random_element(["Blue", "Brown", "Green", "Hazel", "Gray"]),
        "HEIGHT": lambda: f"{fake.random_int(150, 200)} cm",
        "USERAGENT": fake.user_agent,
        "VEHICLEVIN": lambda: fake.bothify("?#?#?#?#?#?#?#?#?").upper(),
        "VEHICLEVRM": fake.license_plate,
        "PHONEIMEI": lambda: fake.numerify("###############"),
        "BITCOINADDRESS": lambda: "1" + fake.pystr(33, 33),
        "ETHEREUMADDRESS": lambda: "0x" + fake.hexify("^" * 40),
        "LITECOINADDRESS": lambda: "L" + fake.pystr(33, 33),
        "MASKEDNUMBER": lambda: fake.numerify("############"),
        "ORDINALDIRECTION": lambda: fake.random_element(["North", "East", "South", "West"]),
        "MEDICALRECORD": lambda: fake.bothify("MRN-########"),
    }


def shape(value: str) -> str:
    """Surface form: letters -> a/A, digits -> 9, other characters kept (``Ab99-9``)."""
    out = re.sub(r"[a-z]", "a", value)
    out = re.sub(r"[A-Z]", "A", out)
    return re.sub(r"\d", "9", out)


def _similar(candidate: str, other: str) -> bool:
    if other == candidate:
        return False
    tolerance = max(1, round(LENGTH_TOLERANCE * len(candidate)))
    if abs(len(other) - len(candidate)) > tolerance:
        return False
    # Same casing and digit layout where the value is structured (codes, numbers, IDs).
    digits = sum(c.isdigit() for c in candidate)
    if digits >= len(candidate) / 2:
        return shape(other) == shape(candidate)
    return candidate[:1].isupper() == other[:1].isupper()


class ReferenceSampler:
    """Draws type-consistent, surface-matched alternatives for candidate values."""

    def __init__(
        self, records: Iterable[Record] = (), source: str = "synthetic", seed: int = 0
    ) -> None:
        if source not in {"synthetic", "corpus", "mix"}:
            raise ValueError("source must be synthetic, corpus, or mix")
        self.source = source
        self.rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
        self.fake = Faker()
        self.fake.seed_instance(seed)
        self.generators = _generators(self.fake)
        self.pool: dict[str, list[str]] = defaultdict(list)
        self.seen: set[str] = set()
        for record in records:
            for span in record.entities:
                value = record.value(span)
                self.pool[span.entity_type].append(value)
                self.seen.add(value)

    def _synthetic(self, entity_type: str, candidate: str) -> str:
        generate = self.generators.get(entity_type)
        best = None
        for _ in range(MAX_DRAWS):
            value = generate() if generate else self._perturb(candidate)
            if value in self.seen or value == candidate:
                continue  # synthetic references must be values the model never trained on
            best = value
            if _similar(candidate, value):
                return value
        return best if best is not None else self._perturb(candidate)

    def _perturb(self, value: str) -> str:
        """A same-shape value for types without a generator: random letters and digits."""
        out = []
        for ch in value:
            if ch.isdigit():
                out.append(str(self.rng.randint(0, 9)))
            elif ch.isalpha():
                letter = self.rng.choice("abcdefghijklmnopqrstuvwxyz")
                out.append(letter.upper() if ch.isupper() else letter)
            else:
                out.append(ch)
        return "".join(out)

    def _from_corpus(self, entity_type: str, candidate: str) -> str | None:
        pool = [v for v in self.pool.get(entity_type, []) if v != candidate]
        if not pool:
            return None
        matched = [v for v in pool if _similar(candidate, v)]
        return self.rng.choice(matched or pool)

    def one(self, entity_type: str, candidate: str) -> str:
        use_corpus = self.source == "corpus" or (self.source == "mix" and self.rng.random() < 0.5)
        if use_corpus:
            value = self._from_corpus(entity_type, candidate)
            if value is not None:
                return value
        return self._synthetic(entity_type, candidate)

    def sample(self, entity_type: str, candidate: str, n: int) -> list[str]:
        """``n`` distinct alternatives (fewer only if the type cannot produce enough)."""
        values: list[str] = []
        for _ in range(n * 4):
            value = self.one(entity_type, candidate)
            if value not in values and value != candidate:
                values.append(value)
            if len(values) == n:
                break
        return values
