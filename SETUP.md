# Setup guide

This guide takes you from nothing installed to a signed model passport, step by step. For what Model Passport is and why it exists, see the [README](README.md).

**Contents**

1. [Install](#1-install)
2. [Run the demo](#2-run-the-demo)
3. [Use your own data](#3-use-your-own-data)
4. [Check new data](#4-check-new-data)
5. [Dashboard and registry](#5-dashboard-and-registry)
6. [Run everything with Docker](#6-run-everything-with-docker)
7. [Block unsafe models in CI](#7-block-unsafe-models-in-ci)
8. [Command reference](#8-command-reference)
9. [How it works](#9-how-it-works)
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

### Install

To use Model Passport on your own data, install the package:

```bash
python3 -m venv .venv                 # a private Python environment
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install "model-passport[dashboard] @ git+https://github.com/r7bb/Model-Passport"
```

After the first release on PyPI, this becomes `pip install "model-passport[dashboard]"`. A ready-made Docker image will be published at `ghcr.io/r7bb/model-passport`.

To run the demo or change the code, clone the repository instead:

```bash
git clone https://github.com/r7bb/Model-Passport.git
cd Model-Passport
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[demo,registry,dashboard]"
```

Check that it worked:

```bash
passport --version
passport --help           # commands, grouped by step
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

## 3. Use your own data

### The quick way: a data file and a label

Put your data file (CSV, TSV, Parquet, or JSONL) in a new folder and name the column to predict:

```bash
mkdir my-model && cd my-model && git init
cp ~/Downloads/customers.csv .
passport init --data customers.csv --label churn
```

`init` checks the file and shows what it found before changing anything:

```
customers.csv: 2,000 rows, 11 columns; `churn` is binary-classification
  identifiers to drop: name, email
  quasi-identifiers:   age, zip
  unused columns:      customer_id (row identifier)
```

It writes `passport.yaml`, three stage scripts in `stages/`, a `policy.yaml`, and a signing key. Then:

```bash
passport run        # clean, split, train, evaluate
passport build      # check, sign, and write passport.json and passport.html
```

It works out on its own:

| What | How |
|---|---|
| The task | Text or yes/no labels: classification. Numbers: regression (a whole-number label with at most 10 values is treated as classes). |
| Column types | Numbers stored as text (`"1,234.50"`), dates, and codes with leading zeros (`"02103"`) are read correctly. |
| Columns to ignore | Row IDs, constant or empty columns, and free text are left out; the train stage lists them. |
| Bad rows | Rows with no label, exact duplicates, and classes with fewer than 3 rows are removed and counted. |
| Personal data | Columns the PII scan flags are dropped; quasi-identifiers (age, ZIP, gender, ...) are bucketed. |
| The model | See [How the model is chosen](#how-the-model-is-chosen). |

Settings live in `passport.yaml` under each stage's `params`. The most useful ones:

| Stage | Setting | Default | Meaning |
|---|---|---|---|
| preprocess | `quasi_identifiers` | `auto` | Columns to bucket, or `auto` to guess from column names |
| preprocess | `identifiers` | `[]` | Extra columns to always drop |
| preprocess | `min_k` | `5` | Smallest group of people who share the same quasi-identifier values |
| preprocess | `max_suppression` | `0.01` | Share of records that may be removed to reach `min_k` |
| preprocess | `task` | `auto` | Force `binary-classification`, `multiclass-classification`, or `regression` |
| train | `model` | `auto` | Or force one family: `linear`, `boosting`, `forest` |
| train | `max_gap` | `0.05` | Largest allowed train-validation gap; stops overfit models |
| train | `search` | `full` | `quick` for a smaller, faster search |
| evaluate | `positive` | second class | Which class counts as positive for precision and recall |

Change one for a single run with `--set`, for example `passport run --set train.search=quick`.

### Bring your own scripts

Any stage can be your own script. Each stage is a command plus the files it reads and writes:

```yaml
stages:
  - name: train
    cmd: python train.py
    params: {seed: 7}                      # passed to your script; override with --set
    deps: [data/train.csv]                 # files it reads
    outs: [models/model.pkl]               # files it writes
    metrics: models/metrics.json           # optional: {split: {metric: value}}
```

Read the parameters in your script with `from model_passport.runtime import params`. Describe the model and data under `build:` (see the demo's [passport.yaml](passport.yaml)), and fill in `declared:` (intended use, limitations, owner), which is shown in the report.

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

A raw batch must first be prepared the same way as the training data: the same identifier columns dropped and the same buckets applied. `passport prepare` does this using the rules the passport certifies, and refuses if they changed:

```bash
python demo/make_dataset.py --rows 1000 --seed 8 --drift 0.7 --out data/batches/week1.csv
passport prepare data/batches/week1.csv          # writes data/batches/week1.prepared.csv
passport monitor drift data/batches/week1.prepared.csv
```

Numbers stored as text are compared as numbers and dates as dates; ID and free-text columns are skipped.

Tuning options: `--alpha` (significance level, default 0.01), `--psi-threshold` (default 0.1), `--tolerance` (allowed accuracy drop, default 0.05).

---

## 5. Dashboard and registry

### Dashboard on your own machine

```bash
passport dashboard
```

This opens a browser page showing the local `passport.json` and its history. To link to a specific version, add `?passport=<passport_id>` to the address.

### Registry: a shared store for passports

The registry is a small server that keeps passports from many models and re-checks each one on upload:

```bash
passport serve --db registry.db                                   # starts at http://localhost:8000
passport push passport.json --registry http://localhost:8000      # upload
passport dashboard --registry http://localhost:8000
```

Interactive API docs are at http://localhost:8000/docs. The main endpoints:

- `POST /passports` and `GET /passports`
- `GET /passports/{id}`, plus `/verify`, `/lineage`, `/dag`, `/html`, and `/jsonld`
- `POST /passports/{id}/events`

To require a password (bearer token) for uploads, set `PASSPORT_REGISTRY_TOKEN` on both the server and the client. To accept only passports signed by known keys, put their `.pub` files in a folder and pass `--trusted-keys <folder>`.

---

## 6. Run everything with Docker

Docker starts the dashboard, the registry, MLflow, and SeaweedFS (S3-compatible file storage for MLflow) together, with nothing else to install:

```bash
cp .env.example .env        # then open .env and replace every password
docker compose up -d --build
```

| Service | Address |
|---|---|
| Dashboard | http://localhost:8501 |
| Registry API | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Object store (S3 API) | http://localhost:8333 |

To log runs to MLflow, set `tracking.mlflow_uri: http://localhost:5000` in `passport.yaml`. Stop everything with `docker compose down`.

---

## 7. Block unsafe models in CI

Use the Model Passport GitHub Action. It runs your pipeline, builds and verifies the passport, and uploads it with its report. The job fails when the verdict is FAIL, and the job summary lists every rule:

```yaml
jobs:
  passport:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: r7bb/Model-Passport@v0.2.0
        with:
          signing-key: ${{ secrets.PASSPORT_SIGNING_KEY }}   # contents of .passport/signing_key.pem
```

| Input | Default | Meaning |
|---|---|---|
| `command` | `build` | `build` (run, build, verify), `build-only`, or `verify` |
| `working-directory` | `.` | Folder with `passport.yaml` |
| `signing-key` | none | Your private key from a repository secret. Without it, a throwaway key is used, which is fine for pull-request checks. |
| `key-passphrase` | none | For an encrypted key |
| `extras` | none | Extra packages, e.g. `mlflow` |
| `python-version` | `3.11` | |
| `upload-artifact` | `true` | Upload `passport.json`, `passport.html`, and history |

Outputs: `verdict` and `passport-id`. Other CI systems can run the same commands directly: `passport build` and `passport verify` exit with an error on failure.

---

## 8. Command reference

Run `passport <command> --help` for every option.

| Command | What it does |
|---|---|
| `passport init [--name]` | Creates `passport.yaml`, a signing key, and `.gitignore` entries |
| `passport init --data <file> --label <column>` | The same, plus ready-made stages for that dataset |
| `passport run [--set stage.key=value]` | Runs the stages and records what they used and produced |
| `passport build [--reason] [--no-link] [--no-html]` | Runs all checks, applies the policy, signs, and writes the report |
| `passport verify [passport.json]` | Checks file hashes, signature, key, and event history |
| `passport report` | Re-creates the HTML report |
| `passport export` | Exports JSON-LD (W3C PROV, DCAT, ML Schema) |
| `passport scan data <file>` | Personal information and re-identification risk in one data file |
| `passport scan secrets <paths>` | Passwords and access keys in code and config |
| `passport audit model <model> --members --nonmembers --label` | Memorization test and model file safety |
| `passport audit deps` | Known vulnerabilities in installed packages |
| `passport prepare <batch>` | Prepares a raw batch exactly like the training data |
| `passport monitor drift <batch>` | Checks new data and records the result |
| `passport serve` | Starts the registry |
| `passport push` | Uploads a passport to a registry |

---

## 9. How it works

### How the model is chosen

The train stage tries three kinds of model, simplest first:

1. **Linear:** logistic regression or ridge regression, tuned over 10 regularization strengths.
2. **Gradient-boosted trees:** tuned over learning rate, tree size, leaf size, and L2 penalty, with early stopping.
3. **Random forest:** tuned over leaf size and features per split.

Each one is scored with stratified 5-fold cross-validation. Classification uses log loss and regression uses RMSE; both reward good probabilities, not just right answers. All cleaning, filling of missing values, scaling, and encoding happen inside each fold, so no fold learns anything from its own validation rows.

Two rules pick the winner:

- **No memorizing:** a candidate whose training score beats its validation score by more than `max_gap` is set aside. It would leak who was in the training data and fail the privacy gate.
- **Simplest that's as good:** among the rest, the simplest family whose score is within one standard error of the best wins. Smaller differences are just noise.

The full comparison is saved to `models/selection.json`, and the train stage prints it:

```
binary-classification: chose linear (LogisticRegression), 5-fold CV accuracy 0.8213 (train-validation gap 0.0044)
  why: best cross-validated score with train-validation gap within 0.05
  * linear    score -0.38641   gap 0.0044
    boosting  score -0.4052    gap 0.0397
    forest    score -0.44185   gap 0.0587  (overfits: set aside)
```

On the demo data the linear model reaches 82.9% test accuracy. That is almost exactly the best any model could do: the synthetic labels contain random noise, which caps accuracy at about 83%. On data with curved or interacting patterns, gradient boosting wins instead. For example, the tests include a two-moons dataset where it beats the linear model by over 3 points.

### How personal details are protected

Direct identifiers (names, emails, phone numbers, ID numbers, dates of birth) are dropped. Quasi-identifiers such as age, ZIP code, gender, and marital status are harmless alone but can single people out together, so preprocess makes sure every combination is shared by at least `min_k` people in each file:

1. It starts with sensible groups: 10-year age ranges and the first 3 digits of ZIP codes.
2. If groups are still too small, it coarsens one column at a time: wider ranges, shorter codes, rare categories merged into "other", or, as a last resort, the column blanked. Each step is the one that protects the most people per unit of information lost about what you're predicting. Columns that only identify people give way first, and columns that help predictions keep their detail.
3. It stops when at most 1% of records are still in small groups, removes those few, and splits the data so every group is divided between train and test in proportion.

The preprocess stage prints what it did, for example:

```
  generalized: age (ranges of 20), gender (removed: too identifying for this much data), zip (removed: ...)
```

On an 800-row file with 4 quasi-identifiers, this reaches k-anonymity without removing any records, and test accuracy stays at 82%.

### How verification works

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

pytest                             # about 200 tests, 92% coverage
ruff check . && ruff format --check .
mypy                               # strict mode
```

The HTML report is built with a small typed builder ([src/model_passport/report/html.py](src/model_passport/report/html.py)) instead of a template engine. Every value is escaped unless explicitly marked safe.

### Releasing

1. Update `__version__` in `src/model_passport/__init__.py` and add a section to [CHANGELOG.md](CHANGELOG.md).
2. Push a tag: `git tag v0.2.0 && git push origin v0.2.0`.

The release workflow then builds and checks the package, publishes it to PyPI and the Docker image to `ghcr.io/r7bb/model-passport`, and creates a GitHub release with the changelog notes. It fails early if the tag doesn't match the version.

One-time setup before the first release:
- On PyPI, add a *pending trusted publisher* for project `model-passport` with owner `r7bb`, repository `Model-Passport`, workflow `release.yml`, and environment `pypi`. No token is stored in GitHub.
- In the repository settings, create an environment named `pypi` (optionally with required reviewers).

### Screenshots

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
| `monitor drift` reports schema problems | Prepare the batch first: `passport prepare <batch>` (see [section 4](#4-check-new-data)) |
| A stage stops with "label ... has 1 distinct value" or "only N usable rows" | The data can't train a model yet: check the label column and collect more rows |
| The train stage says "no candidate had a gap within 0.05" | Every model overfits, usually because the data is small. Add rows, or raise `max_gap` knowing the privacy gate may warn |
| k-anonymity shows "not evaluated: no quasi-identifiers found" | List the columns that could identify a person under `privacy.quasi_identifiers` in `passport.yaml`, or ignore the warning if there are none |
| Docker services don't start | Check `docker compose ps` and `docker compose logs <service>`. Make sure `.env` exists and ports 5000, 8000, 8501, 9000, and 9001 are free. |
| The registry rejects uploads with 401 | Set the same `PASSPORT_REGISTRY_TOKEN` for the server and for `passport push` |
