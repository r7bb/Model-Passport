# Model Passport Roadmap

Model Passport extends the AIPassport framework (Kalokyri et al., arXiv 2506.22358) with automated privacy scanning, leakage auditing, artifact safety checks, cryptographic identity, a CI policy gate, and monitoring hooks.

## Decisions taken for the MVP

| Open question | Decision | Rationale |
|---|---|---|
| Tabular only or tabular plus text? | Tabular first. Free text columns are scanned with our regex validators; Presidio is an optional extra (`pip install model-passport[presidio]`). | Keeps the default install light and CI fast. Presidio needs a spaCy model download. |
| Streamlit or React dashboard? | Streamlit | Fastest to build; the registry API stays UI-agnostic, so a React client can be added later. |
| Which stretch goal first? | None in the MVP. Shadow model attacks are next (see Later). | The loss threshold attack already shows the overfit vs regularized contrast the demo needs. |
| HTML report rendering | Typed Python builder (`report/html.py`) instead of Jinja2 | Type-checked and linted like the rest of the code, no template language or dependency, and escaping is still automatic. |
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

### Phase 5: Registry, report, dashboard (done)
- FastAPI and SQLite registry: upload, list, get, verify, and events endpoints, with an optional bearer token for writes.
- Static `passport.html` report rendered by a small typed Python HTML builder (escape-by-default, no template engine), and JSON-LD export using W3C PROV terms.
- Streamlit dashboard: model picker, verdict, lineage, pipeline DAG, scan results, and a plain-language "simple view".
- Docker Compose running MLflow, MinIO (as MLflow's artifact store), the registry, and the dashboard.

### Phase 6: CI and monitoring (done)
- GitHub Actions: lint, tests, and an end-to-end demo run through `passport build` and `passport verify`. CI fails if the demo data carries PII.
- Drift checks (Kolmogorov-Smirnov, Mann-Whitney U, and Cramér-von Mises for numeric features; chi-square for categorical) appended as hash-chained, signed lifecycle events.
- New and changing data: schema validation and live accuracy on each batch, `retrain_recommended` exit code, and `passport build` linking each rebuild to the passport it supersedes (dataset, artifact, and metric deltas; archived history in `.passport/history/`).

## Status and next steps

State at end of session 2 (2026-09-23): all six phases are done. There are 151 tests at about 90% coverage. ruff runs with a strict rule set, mypy --strict is clean, and pre-commit hooks are configured. CI (lint, mypy, tests on 3.11 and 3.12, the full demo lifecycle as a policy gate, and a Docker Compose build with health checks) is green on GitHub Actions.

Session 2 added:
- Tests for the drift, monitor, registry, client, revision, report, and CLI modules. They caught three real bugs, all fixed.
- Demo ML hardening: cross-validated model selection, no-leakage pipelines, fuller metrics, batch preparation from recorded parameters, and `demo/update_cycle.sh`.
- The CI workflow and the Docker Compose stack. MinIO images now come from quay.io.
- The quality pass and refactors.
- Replacing Jinja2 with a typed HTML builder.
- README screenshots from real runs.
- A plain-language README for non-technical readers. Installation, usage, commands, and technical details moved to SETUP.md.

Possible next steps, in rough priority order:
1. Screenshot automation: regenerate `docs/images/` in CI (or with `make docs`) so the README never goes stale.
2. The stretch goals under Later: shadow-model MIA via ML Privacy Meter or ART, Opacus differential privacy in the demo, and text memorization tests.
3. Key management: key rotation, and a trusted-key registry with revocation.
4. A Postgres backend for the registry, plus pagination.
5. Coverage gaps: `registry/client.py` error paths, `core/tracking.py` MLflow failure paths, and `cli/project.py` edge cases.

Environment notes:
- Use `.venv` (Python 3.11 from uv, installed at `~/.local/bin`). Install with `uv pip install --python .venv/bin/python -e ".[dev,demo,registry,dashboard,mlflow,docs]"` and run `python -m playwright install chromium` for screenshots.
- Set `MLFLOW_DISABLE_AGENT_HINT=1` to silence MLflow's startup message.
- Tests stub pip-audit (see `tests/conftest.py`); mark a test `@pytest.mark.network` to use the real one.
- Docker isn't installed on the dev machine, so the compose stack is validated only in CI.
- CI job logs need repo-admin access. On failure the docker job publishes container logs as annotations, which are readable via the public API.
- Commits carry no AI attribution lines.

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
