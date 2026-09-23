"""Prepare an incoming raw batch exactly as the certified model's training data was prepared.

Reads the preprocess parameters recorded in the signed passport (not the current config), so
the batch matches the model the passport describes:

    python demo/prepare_batch.py data/batches/week1.csv --out data/batches/week1.prepared.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from preprocess import DEFAULTS, transform


def recorded_params(passport: Path) -> dict:
    stages = json.loads(passport.read_text(encoding="utf-8")).get("pipeline", [])
    for stage in stages:
        if stage["name"] == "preprocess":
            return {**DEFAULTS, **stage["parameters"]}
    raise SystemExit(f"{passport} has no preprocess stage; run `passport run` and build first")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--passport", type=Path, default=Path("passport.json"))
    args = parser.parse_args()

    p = recorded_params(args.passport)
    batch = pd.read_csv(args.batch, dtype={"zip": str})
    prepared = transform(batch, p["drop_identifiers"], p["generalize"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.out, index=False)
    print(f"prepared {len(prepared)} rows -> {args.out}")


if __name__ == "__main__":
    main()
