"""Preprocess stage: drop direct identifiers, generalize quasi-identifiers, split.

``transform`` is shared with ``prepare_batch.py`` so live batches get exactly the same
treatment as the training data (no train/serve skew).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from model_passport.runtime import params

DIRECT_IDENTIFIERS = ["name", "email", "phone", "ssn", "dob", "notes"]
LABEL = "income"
DEFAULTS = {
    "input": "data/raw.csv",
    "train_out": "data/train.csv",
    "test_out": "data/test.csv",
    "drop_identifiers": True,
    "generalize": True,
    "test_size": 0.25,
    "seed": 7,
}


def age_bucket(age: int) -> str:
    if age < 30:
        return "18-29"
    if age >= 70:
        return "70+"
    low = (age // 10) * 10
    return f"{low}-{low + 9}"


def transform(df: pd.DataFrame, drop_identifiers: bool, generalize: bool) -> pd.DataFrame:
    """Row-wise preprocessing; safe to apply to a single new batch."""
    out = df.copy()
    if drop_identifiers:
        out = out.drop(columns=[c for c in DIRECT_IDENTIFIERS if c in out.columns])
    if generalize:
        out["age"] = out["age"].map(age_bucket)
        out["zip"] = out["zip"].astype(str).str[:3] + "**"
    return out


def main() -> None:
    p = params(DEFAULTS)
    raw = pd.read_csv(p["input"], dtype={"zip": str})
    before = len(raw)
    raw = raw.dropna(subset=[LABEL]).drop_duplicates()
    df = transform(raw, p["drop_identifiers"], p["generalize"])

    # Stratify so both splits keep the label balance; fixed seed for reproducibility.
    train, test = train_test_split(
        df, test_size=p["test_size"], random_state=p["seed"], stratify=df[LABEL]
    )
    for frame, out in ((train, p["train_out"]), (test, p["test_out"])):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
    print(
        f"rows={before} kept={len(df)} train={len(train)} test={len(test)} "
        f"positive_rate={df[LABEL].eq('>50K').mean():.3f} columns={list(df.columns)}"
    )


if __name__ == "__main__":
    main()
