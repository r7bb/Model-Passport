#!/usr/bin/env bash
# End-to-end lifecycle on synthetic data: certify, monitor a drifting batch, retrain, re-certify.
#   bash demo/update_cycle.sh
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n== %s\n' "$*"; }

step "1. Initial data and certified model"
python demo/make_dataset.py --rows 4000 --seed 42
passport init --name income-classifier >/dev/null
passport run
passport build --no-link

step "2. A stable batch arrives: no retraining needed"
python demo/make_dataset.py --rows 1000 --seed 101 --out data/batches/stable.csv
passport prepare data/batches/stable.csv --out data/batches/stable.prepared.csv
passport monitor drift data/batches/stable.prepared.csv

step "3. A shifted batch arrives: drift is detected"
python demo/make_dataset.py --rows 1000 --seed 102 --drift 0.7 --out data/batches/shifted.csv
passport prepare data/batches/shifted.csv --out data/batches/shifted.prepared.csv
if passport monitor drift data/batches/shifted.prepared.csv; then
  echo "expected drift" >&2
  exit 1
fi

step "4. New data is collected and the model is retrained and re-certified"
python demo/make_dataset.py --rows 1500 --seed 103 --drift 0.7 --append data/raw.csv
passport run
passport build --reason "Retrained after drift in age, education, hours_per_week"
passport verify

step "5. Recheck the shifted batch against the retrained model"
# Live accuracy should now match the new test baseline. Drift may still be flagged: the batch is
# entirely the new population while the reference mixes old and new data.
passport monitor drift data/batches/shifted.prepared.csv || true
echo "history: $(ls .passport/history | wc -l | tr -d ' ') archived passport(s)"
