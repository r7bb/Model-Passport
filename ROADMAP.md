# Model Passport Roadmap

Model Passport extends the AIPassport framework (Kalokyri et al., arXiv 2506.22358) with automated privacy scanning, leakage auditing, artifact safety checks, cryptographic identity, a CI policy gate, and monitoring hooks.

## Decisions taken for the MVP

| Open question | Decision | Rationale |
|---|---|---|
| Tabular only or tabular plus text? | Tabular first. Free text columns are scanned with our regex validators; Presidio is an optional extra (`pip install model-passport[presidio]`). | Keeps the default install light and CI fast. Presidio needs a spaCy model download. |
| Streamlit or React dashboard? | Streamlit | Fastest to build; the registry API stays UI-agnostic, so a React client can be added later. |
| Which stretch goal first? | None in the MVP. Shadow model attacks are next (see Later). | The loss threshold attack already shows the overfit vs regularized contrast the demo needs. |
| Model formats for leakage audit | scikit-learn estimators saved with pickle/joblib, loaded only after a pickle safety scan | Matches the demo; ONNX and safetensors are recognized as safe formats by the artifact scanner. |

## Phases

### Phase 1: Core passport and identity (done)
- Pydantic schema v0.1, streamed SHA256, domain-separated Merkle root, canonical JSON, Ed25519 signing.
- `passport init`, `passport build`, `passport verify`, which names every changed or missing file.

### Phase 2: Provenance capture (done)
- `stages:` in `passport.yaml`; `passport run` executes each stage and records command, script hash, git commit and dirty state, parameters, input and output hashes, and timings.
- Parameter overrides with `passport run --set stage.key=value`, passed to scripts through `PASSPORT_PARAMS`.
- Environment capture: Python, OS, hardware, installed packages.
- Optional MLflow logging (one run per stage, passport attached at build) and DVC tracking of stage outputs.

### Phase 3: Data scanners and policy (done)
- Common `Scanner` interface returning `Finding` objects (counts, categories, masked examples only).
- PII scanner: column name heuristics plus validators for email, phone, SSN, Luhn-checked credit cards, IP addresses, and dates of birth. Sampling with an optional full scan.
- Reidentification risk: k anonymity, l diversity, unique record fraction, and the smallest column combinations that make records unique.
- Secrets scanner: known token patterns plus Shannon entropy, over data files and pipeline scripts.
- Policy engine driven by `policy.yaml`, with pass, warn, and fail per rule. `passport build` exits nonzero on fail.

### Phase 4: Model auditors (done)
- Loss threshold membership inference: attack AUC and TPR at 1% FPR.
- Generalization gap between train and test metrics.
- Pickle opcode scan (picklescan); warns when raw pickle is used instead of safetensors or ONNX.
- `pip-audit` of the captured environment, with CVE severity from OSV.

### Phase 5: Registry, report, dashboard (built; tests and Docker Compose pending)
- FastAPI and SQLite registry: upload, list, get, verify, and events endpoints, with an optional bearer token for writes.
- Static `passport.html` report rendered with Jinja2, and JSON-LD export using W3C PROV terms.
- Streamlit dashboard: model picker, verdict, lineage, pipeline DAG, scan results, and a plain-language "simple view".
- Docker Compose running MLflow, MinIO (as MLflow's artifact store), the registry, and the dashboard.

### Phase 6: CI and monitoring (monitoring built; tests and CI pending)
- GitHub Actions: lint, tests, and an end-to-end demo run through `passport build` and `passport verify`. CI fails if the demo data carries PII.
- Drift checks (Kolmogorov-Smirnov, Mann-Whitney U, and Cramér-von Mises for numeric features; chi-square for categorical) appended as hash-chained, signed lifecycle events.
- New and changing data: schema validation and live accuracy on each batch, `retrain_recommended` exit code, and `passport build` linking each rebuild to the passport it supersedes (dataset, artifact, and metric deltas; archived history in `.passport/history/`).

## Next session: pick up here

State at end of session 1 (2026-09-22): Phases 1-4 are done and tested (110 tests passing). Phase 5 and 6 code is written and was smoke-tested end to end by hand (monitor on stable vs drifted batches, retrain and revision linking, registry upload/verify/lineage, dashboard against the registry), but it has **no unit tests yet**. Work in this order:

1. **Tests for the untested modules:** `core/events.py` (partly covered in `test_build_verify.py`), `core/revision.py`, `core/jsonld.py`, `monitoring/drift.py` (stable vs shifted data, Bonferroni, PSI gate, schema checks), `monitoring/monitor.py` (reference hash check, key mismatch, performance degradation), `registry/api.py` via `fastapi.testclient` (upload rejects tampered passports, 409 duplicates, token auth, event chain rejection, lineage), `registry/sources.py`, `report/render.py` and `report/summary.py` (autoescaping, no raw PII), and the new CLI commands (`monitor drift`, `push`, `report`, `export`, `audit`).
2. **Demo ML hardening** (requested): cross-validated model selection in `demo/train.py` (GridSearchCV over `C` with StratifiedKFold, CV score in `model_info.json`); more metrics in `demo/evaluate.py` (log loss, precision, recall, Brier); a `demo/prepare_batch.py` that applies the same preprocessing to incoming batches (today this is done inline in the README flow); a `demo/update_cycle.sh` covering monitor, then append, then run, then build.
3. **Phase 6 CI:** `.github/workflows/ci.yml` running ruff, mypy, and pytest, then the demo (`make_dataset`, `init`, `run`, `build`, `verify`). A push that adds a PII column to the demo data must fail the build.
4. **Docker Compose** (`docker-compose.yml`, `Dockerfile`): MLflow with MinIO as the artifact store, the registry, the dashboard, and a `.env.example` (never commit `.env`). Docker is not installed on the dev machine, so validate in CI.
5. **Quality pass** (requested: "proper checkstyle, no smelly code, good practices"): widen ruff rules (C90 complexity, N, PL, PTH, RET, ARG, ERA, S, BLE, TRY), add `mypy --strict`, add `.pre-commit-config.yaml`, remove the `dashboard/app.py` E501 ignore, and refactor what they flag (the long `build_passport` and dashboard functions first).
6. **README images:** more screenshots via `scripts/screenshot.py` (dashboard Pipeline, Lineage, Privacy, and Monitoring tabs; the FAIL report), plus CLI output captures for build FAIL/PASS, verify tamper, and monitor drift.

Environment notes: use `.venv` (Python 3.11 from uv, installed at `~/.local/bin`); install with `uv pip install --python .venv/bin/python -e ".[dev,demo,registry,dashboard,mlflow,docs]"`. Set `MLFLOW_DISABLE_AGENT_HINT=1` to silence MLflow's startup message. Tests stub pip-audit (see `tests/conftest.py`); mark a test `@pytest.mark.network` to use the real one. Commits carry no AI attribution lines.

Deviations from the original project spec to confirm with the team:
- Passports carry a top-level `artifacts` manifest plus `run` and `revision` sections that the v0.1 spec did not list.
- `unsafe_pickle` defaults to `fail` (not `warn`) for pickles importing dangerous callables. Plain raw pickle is a low-severity finding, so it doesn't block.
- A rule with no evidence (e.g. no leakage audit) yields `missing_evidence: warn`, not a pass.

## Later
- Shadow model membership inference via ML Privacy Meter or IBM ART.
- Text model memorization: canary exposure tests and PII extraction prompts.
- Differential privacy training (Opacus) in the demo.
- Key rotation, and a trusted key registry with revocation.
- Postgres backend for the registry; React dashboard.
- Signed model cards exported to Hugging Face and MLflow model registry tags.
- DCAT and ML Schema alignment alongside PROV in the JSON-LD export.
