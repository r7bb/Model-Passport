"""Generate a synthetic census-style income dataset with injected fake PII.

All values are synthetic (numpy + Faker). Nothing here describes a real person.

    python demo/make_dataset.py --rows 4000 --out data/raw.csv
    python demo/make_dataset.py --rows 1000 --seed 7 --drift 0.6 --out data/batches/week1.csv
    python demo/make_dataset.py --rows 1000 --seed 8 --drift 0.6 --append data/raw.csv

``--drift`` (0 to 1) simulates a population shift in new data: an older, more educated,
longer-working population. ``--append`` adds the rows to an existing file (new data arriving).
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

EDUCATION = ["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate", "Assoc"]
OCCUPATION = ["Tech", "Sales", "Service", "Craft", "Admin", "Professional", "Transport"]
MARITAL = ["Married", "Never-married", "Divorced", "Widowed"]
# A small pool of ZIP codes (five 3-digit prefixes) so generalization can reach k-anonymity.
ZIP_PREFIXES = ["021", "100", "303", "606", "941"]


BASE_EDUCATION_P = np.array([0.3, 0.2, 0.25, 0.12, 0.03, 0.1])
SHIFTED_EDUCATION_P = np.array([0.1, 0.15, 0.3, 0.3, 0.08, 0.07])


def make(rows: int, seed: int, drift: float = 0.0) -> pd.DataFrame:
    if not 0.0 <= drift <= 1.0:
        raise ValueError("drift must be between 0 and 1")
    rng = np.random.default_rng(seed)
    fake = Faker("en_US")
    Faker.seed(seed)

    age = rng.integers(18 + round(22 * drift), 80, rows)
    education_p = (1 - drift) * BASE_EDUCATION_P + drift * SHIFTED_EDUCATION_P
    education = rng.choice(EDUCATION, rows, p=education_p / education_p.sum())
    occupation = rng.choice(OCCUPATION, rows)
    marital = rng.choice(MARITAL, rows, p=[0.45, 0.35, 0.15, 0.05])
    gender = rng.choice(["F", "M"], rows)
    zip5 = [f"{rng.choice(ZIP_PREFIXES)}{rng.integers(0, 8):02d}" for _ in range(rows)]
    hours = np.clip(rng.normal(40 + 8 * drift, 10, rows).round(), 5, 90).astype(int)
    capital_gain = np.where(rng.random(rows) < 0.08, rng.integers(1000, 20000, rows), 0)

    edu_score = pd.Series(education).map(
        {
            "HS-grad": 0,
            "Assoc": 0.5,
            "Some-college": 0.4,
            "Bachelors": 1.2,
            "Masters": 1.8,
            "Doctorate": 2.2,
        }
    )
    logit = (
        -4.0
        + 0.045 * np.minimum(age, 60)
        + 0.9 * edu_score.to_numpy()
        + 0.03 * (hours - 40)
        + 0.8 * (marital == "Married")
        + 0.00012 * capital_gain
        + 0.5 * np.isin(occupation, ["Tech", "Professional"])
        + rng.normal(0, 0.8, rows)
    )
    income = np.where(1 / (1 + np.exp(-logit)) > 0.5, ">50K", "<=50K")

    today = date(2026, 1, 1)
    df = pd.DataFrame(
        {
            # Direct identifiers (fake, injected for the demo).
            "name": [fake.name() for _ in range(rows)],
            "email": [fake.email() for _ in range(rows)],
            "phone": [fake.phone_number() for _ in range(rows)],
            "ssn": [fake.ssn() for _ in range(rows)],
            "dob": [
                fake.date_between_dates(
                    date(today.year - int(a) - 1, today.month, today.day),
                    date(today.year - int(a), today.month, today.day),
                ).isoformat()
                for a in age
            ],
            "notes": [
                f"Follow up at {fake.email()}" if rng.random() < 0.1 else "No notes."
                for _ in range(rows)
            ],
            # Quasi-identifiers and features.
            "age": age,
            "gender": gender,
            "zip": zip5,
            "education": education,
            "occupation": occupation,
            "marital_status": marital,
            "hours_per_week": hours,
            "capital_gain": capital_gain,
            "income": income,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift", type=float, default=0.0, help="population shift, 0 to 1")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--out", type=Path, default=Path("data/raw.csv"))
    target.add_argument("--append", type=Path, help="append rows to an existing dataset")
    args = parser.parse_args()

    frame = make(args.rows, args.seed, args.drift)
    if args.append is not None:
        existing = pd.read_csv(args.append, dtype={"zip": str})
        pd.concat([existing, frame], ignore_index=True).to_csv(args.append, index=False)
        print(f"appended {args.rows} rows (drift={args.drift}) to {args.append}")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"wrote {args.rows} synthetic rows (drift={args.drift}) to {args.out}")


if __name__ == "__main__":
    main()
