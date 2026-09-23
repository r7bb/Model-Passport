# Setup guide

This guide takes you from nothing installed to a signed model passport, step by step. For what Model Passport is and why it exists, see the [README](README.md).

**Contents**

1. [Install](#1-install)
2. [Run the demo](#2-run-the-demo)
3. [Use it on your own project](#3-use-it-on-your-own-project)
4. [Check new data](#4-check-new-data)
5. [Dashboard and registry](#5-dashboard-and-registry)
6. [Run everything with Docker](#6-run-everything-with-docker)
7. [Block unsafe models in CI](#7-block-unsafe-models-in-ci)
8. [Command reference](#8-command-reference)
9. [How verification works](#9-how-verification-works)
10. [Contributing](#10-contributing)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Install

### What you need

| Tool | Why | How to get it |
|---|---|---|
| Python 3.11 or newer | Model Passport is written in Python | [python.org/downloads](https://www.python.org/downloads/), or `brew install python@3.11` on macOS |
| Git | To download the code and record which code version trained a model | [git-scm.com](https://git-scm.com/downloads) |
| Docker (optional) | Only for [section 6](#6-run-everything-with-docker) | [docker.com](https://www.docker.com/products/docker-desktop/) |

Check your Python version. It must print 3.11 or higher:

```bash
python3 --version
```

### Download and install

```bash
git clone https://github.com/r7bb/Model-Passport.git
cd Model-Passport

python3 -m venv .venv                 # a private Python environment for this project
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -e ".[demo,registry,dashboard]"
```

Check that it worked:

```bash
passport --help
```

You should see a list of commands. Run `source .venv/bin/activate` again whenever you open a new terminal.

### Optional add-ons

Add any of these inside the brackets, separated by commas, for example `pip install -e ".[demo,mlflow]"`:

| Extra | Adds |
|---|---|
| `demo` | Fake-data generator used by the demo |
| `registry` | A server that stores and re-checks passports |
| `dashboard` | The web dashboard |
| `mlflow` | Logging runs to MLflow |
| `presidio` | Deeper personal-information detection in free text |
| `dev` | Tests, linters, and type checking (for contributors) |
| `docs` | Screenshot tools used to make the README images |

---

## 2. Run the demo

The demo trains a model that predicts income from made-up census-style records. The data includes fake names, emails, and phone numbers so the checks have something to find. No real people are involved.

### The quick way

On macOS or Linux, one command runs the whole story:

```bash
bash demo/update_cycle.sh
```

It creates the data, trains and certifies a model, checks two new batches of data (one similar, one different), retrains, and re-certifies. On Windows, run the steps below instead.

### Step by step

**1. Create the fake data and a signing key.**

```bash
python demo/make_dataset.py
passport init
```

`passport init` creates a private signing key in `.passport/`. Git ignores this folder; never share or commit it.

**2. Train an unsafe model and watch it fail.**

This keeps personal details in the data and lets the model memorize its training records:

```bash
passport run --set preprocess.drop_identifiers=false \
             --set preprocess.generalize=false \
             --set train.model=overfit
passport build
```

The result is **FAIL**, with the reasons listed. Open `passport.html` in a browser to read the report.

**3. Train the safe version and watch it pass.**

This removes identifiers, groups ages and ZIP codes into ranges, and tunes the model so it generalizes:

```bash
passport run
passport build
passport verify
```

The result is **PASS**, and `verify` confirms nothing has changed since signing.

**4. Try tampering.**

Change any certified file and verify again:

```bash
echo "tampered" >> models/model_info.json
passport verify          # names the changed file and exits with an error
passport run && passport build   # rebuild to get back to a clean state
```

---

## 3. Use it on your own project

### Create the config

In your project folder:

```bash
passport init --name my-model
```

This creates `passport.yaml` (describes your pipeline) and a signing key in `.passport/`. Copy [policy.yaml](policy.yaml) from this repository to set pass and fail limits.

### Describe your pipeline

Edit `passport.yaml`. Each **stage** is a command you already run, plus the files it reads and writes:

```yaml
project:
  name: my-model
  version: 1.0.0

signing:
  private_key: .passport/signing_key.pem   # never commit this file
  public_key: .passport/signing_key.pub

stages:
  - name: preprocess
    cmd: python preprocess.py
    params: {seed: 7}                      # passed to your script; override with --set
    deps: [data/raw.csv]                   # files it reads
    outs: [data/train.csv, data/test.csv]  # files it writes

  - name: train
    cmd: python train.py
    deps: [data/train.csv]
    outs: [models/model.pkl]

  - name: evaluate
    cmd: python evaluate.py
    deps: [models/model.pkl, data/test.csv]
    outs: [models/metrics.json]
    metrics: models/metrics.json           # scores shown in the passport

declared:                                  # written by a person, shown in the report
  intended_use: What the model is for.
  out_of_scope_uses: [Uses it must not be put to.]
  known_limitations: [Where it may be wrong.]
  owner: Your team
```

See the demo's [passport.yaml](passport.yaml) and scripts in [demo/](demo/) for a complete working example.

### Run, build, verify

```bash
passport run        # runs each stage and records scripts, settings, files, and the git commit
passport build      # runs the checks, applies policy.yaml, signs; writes passport.json and passport.html
passport verify     # confirms nothing changed since signing
```

`passport build` exits with an error when the verdict is FAIL.

### Adjust the policy

[policy.yaml](policy.yaml) sets the limits. Each rule is either a single fail limit, or a warn and fail pair:

```yaml
rules:
  pii_columns_max: 0                          # columns with personal identifiers
  min_k_anonymity: 5                          # every combination of age/ZIP/etc. shared by 5+ people
  unique_record_fraction_max: 0.05
  mia_auc_max: {warn: 0.55, fail: 0.60}       # how well an attacker can tell who was in the training data
  generalization_gap_max: {warn: 0.05, fail: 0.10}  # train score minus test score
  secrets_found_max: 0
  unsafe_pickle: fail
  critical_cves_max: 0
missing_evidence: warn                        # verdict when a check could not run
```

### Retraining and versions

When you retrain, run `passport build` again. The new passport links to the one it replaces, and the old one is archived in `.passport/history/`. Add a note to explain why:

```bash
passport build --reason "Retrained on September data"
```

Use `--no-link` to start a fresh history instead.

---

## 4. Check new data

When new data arrives, check whether it still looks like the training data:

```bash
passport monitor drift data/new_batch.csv
```

It checks three things:

- **Shape:** the same columns and types as the training data.
- **Distribution:** whether each column has shifted. It uses KS, Mann-Whitney U, and Cramér-von Mises tests for numbers and chi-square for categories, corrected for testing many columns at once. A shift only counts if its PSI is at least 0.1.
- **Accuracy:** if the batch includes the true answers, whether accuracy dropped by more than 5 points.

The result is added to the passport as a signed event. The command exits with an error when retraining is recommended, so you can run it on a schedule.

The batch must be preprocessed the same way as the training data. The demo shows how in [demo/prepare_batch.py](demo/prepare_batch.py):

```bash
python demo/make_dataset.py --rows 1000 --seed 8 --drift 0.7 --out data/batches/week1.csv
python demo/prepare_batch.py data/batches/week1.csv --out data/batches/week1.prepared.csv
passport monitor drift data/batches/week1.prepared.csv
```

Tuning options: `--alpha` (significance level, default 0.01), `--psi-threshold` (default 0.1), `--tolerance` (allowed accuracy drop, default 0.05).

---

## 5. Dashboard and registry

### Dashboard on your own machine

```bash
streamlit run dashboard/app.py
```

This opens a browser page showing the local `passport.json` and its history. To link to a specific version, add `?passport=<passport_id>` to the address.

### Registry: a shared store for passports

The registry is a small server that keeps passports from many models and re-checks each one on upload:

```bash
passport serve --db registry.db                                   # starts at http://localhost:8000
passport push passport.json --registry http://localhost:8000      # upload
PASSPORT_REGISTRY_URL=http://localhost:8000 streamlit run dashboard/app.py
```

Interactive API docs are at http://localhost:8000/docs. The main endpoints:

- `POST /passports` and `GET /passports`
- `GET /passports/{id}`, plus `/verify`, `/lineage`, `/dag`, `/html`, and `/jsonld`
- `POST /passports/{id}/events`

To require a password (bearer token) for uploads, set `PASSPORT_REGISTRY_TOKEN` on both the server and the client. To accept only passports signed by known keys, put their `.pub` files in a folder and pass `--trusted-keys <folder>`.

---

## 6. Run everything with Docker

Docker starts the dashboard, the registry, MLflow, and MinIO (file storage for MLflow) together, with nothing else to install:

```bash
cp .env.example .env        # then open .env and replace every password
docker compose up -d --build
```

| Service | Address |
|---|---|
| Dashboard | http://localhost:8501 |
| Registry API | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| MinIO console | http://localhost:9001 |

To log runs to MLflow, set `tracking.mlflow_uri: http://localhost:5000` in `passport.yaml`. Stop everything with `docker compose down`.

---

## 7. Block unsafe models in CI

Because `passport build` and `passport verify` exit with an error on failure, any CI system can use them as a gate. A minimal GitHub Actions job:

```yaml
jobs:
  passport:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with: {python-version: "3.11"}
      - run: pip install "model-passport @ git+https://github.com/r7bb/Model-Passport"
      - run: passport scan secrets src
      - run: passport init              # CI key; use a stored secret key for real releases
      - run: passport run
      - run: passport build             # fails the job on a FAIL verdict
      - run: passport verify passport.json
```

This repository's own pipeline is in [.github/workflows/ci.yml](.github/workflows/ci.yml).

---

## 8. Command reference

Run `passport <command> --help` for every option.

| Command | What it does |
|---|---|
| `passport init [--name]` | Creates `passport.yaml`, a signing key, and `.gitignore` entries |
| `passport run [--set stage.key=value]` | Runs the stages and records what they used and produced |
| `passport build [--reason] [--no-link] [--no-html]` | Runs all checks, applies the policy, signs, and writes the report |
| `passport verify [passport.json]` | Checks file hashes, signature, key, and event history |
| `passport report` | Re-creates the HTML report |
| `passport export` | Exports JSON-LD (W3C PROV, DCAT, ML Schema) |
| `passport scan data <file>` | Personal information and re-identification risk in one data file |
| `passport scan secrets <paths>` | Passwords and access keys in code and config |
| `passport audit model <model> --members --nonmembers --label` | Memorization test and model file safety |
| `passport audit deps` | Known vulnerabilities in installed packages |
| `passport monitor drift <batch>` | Checks new data and records the result |
| `passport serve` | Starts the registry |
| `passport push` | Uploads a passport to a registry |

---

## 9. How verification works

- **Fingerprints:** every file (config, policy, model, data, scripts, stage inputs and outputs) gets a SHA256 hash. Changing even one byte changes its hash.
- **One summary hash:** the file hashes are combined into a Merkle root, so a single value covers every file.
- **Signature:** the passport is signed with an Ed25519 private key. Anyone with the public key can confirm it hasn't been edited.
- **Event history:** results from later checks, such as drift monitoring, are signed separately and chained together. Events can be added but not edited, reordered, or moved to another passport.
- **No sensitive values:** findings contain only counts, rates, and masked examples such as `j***@e***.com`, never the actual personal data.

### What the checks measure

- **Personal information:** pattern and column-name matching for emails, phones, national IDs, card numbers, and more. Presidio is optional for free text.
- **Re-identification risk:** k-anonymity, l-diversity, and the share of unique records over quasi-identifiers such as age, ZIP code, and sex.
- **Memorization:** a loss-threshold membership inference attack (AUC and true-positive rate at 1% false positives) plus the train-test gap.
- **File safety:** picklescan for dangerous imports in model files, and pip-audit with OSV severities for dependencies.

---

## 10. Contributing

```bash
pip install -e ".[dev,demo,registry,dashboard]"
pre-commit install                 # runs lint, type checks, and a secrets scan on every commit

pytest                             # 150+ tests, about 90% coverage
ruff check . && ruff format --check .
mypy                               # strict mode
```

The HTML report is built with a small typed builder ([src/model_passport/report/html.py](src/model_passport/report/html.py)) instead of a template engine. Every value is escaped unless explicitly marked safe.

To regenerate the README images from real runs:

```bash
pip install -e ".[docs]" && python -m playwright install chromium
python scripts/screenshot.py --help
python scripts/terminal_shot.py --help
```

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `passport: command not found` | Activate the environment: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`) |
| `python3 --version` shows 3.9 or 3.10 | Install Python 3.11+, then create the environment with it: `python3.11 -m venv .venv` |
| `passport init` says "kept existing" | That's expected: it never overwrites your config or key. `--force` replaces both, but older passports then need the old public key to verify. |
| `verify` reports a changed file you didn't mean to change | Rerun `passport run && passport build` to certify the current files |
| `monitor drift` reports schema problems | Prepare the batch with the same preprocessing as training (see [section 4](#4-check-new-data)) |
| Docker services don't start | Check `docker compose ps` and `docker compose logs <service>`. Make sure `.env` exists and ports 5000, 8000, 8501, 9000, and 9001 are free. |
| The registry rejects uploads with 401 | Set the same `PASSPORT_REGISTRY_TOKEN` for the server and for `passport push` |
