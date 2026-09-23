"""Preprocess stage: optionally drop direct identifiers and generalize quasi-identifiers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from model_passport.runtime import params

DIRECT_IDENTIFIERS = ["name", "email", "phone", "ssn", "dob", "notes"]


def age_bucket(age: int) -> str:
    if age < 30:
        return "18-29"
    if age >= 70:
        return "70+"
    low = (age // 10) * 10
    return f"{low}-{low + 9}"


def main() -> None:
    p = params(
        {
            "input": "data/raw.csv",
            "train_out": "data/train.csv",
            "test_out": "data/test.csv",
            "drop_identifiers": True,
            "generalize": True,
            "test_size": 0.25,
            "seed": 7,
        }
    )
    df = pd.read_csv(p["input"], dtype={"zip": str})
    if p["drop_identifiers"]:
        df = df.drop(columns=[c for c in DIRECT_IDENTIFIERS if c in df.columns])
    if p["generalize"]:
        df["age"] = df["age"].map(age_bucket)
        df["zip"] = df["zip"].str[:3] + "**"

    train, test = train_test_split(
        df, test_size=p["test_size"], random_state=p["seed"], stratify=df["income"]
    )
    for frame, out in ((train, p["train_out"]), (test, p["test_out"])):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
    print(f"train={len(train)} test={len(test)} columns={list(df.columns)}")


if __name__ == "__main__":
    main()
