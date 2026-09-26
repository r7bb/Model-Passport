# MP roadmap

MP (Model Passport) is an entity-level PII leakage auditing and remediation platform: it finds personal data a model memorized, removes it, verifies the fix, and records every step in a signed passport. It builds on EL-MIA (Satvaty et al., LREC 2026) and the AIPassport framework (Kalokyri et al., arXiv 2506.22358).

## Where things stand

The passport engine (phases 1–7) and platform phases A–D are done: 246 Python tests plus Go tests, strict ruff and mypy, and green CI. CI covers lint, tests on Python 3.11 and 3.12, the demo lifecycle, the wheel, the GitHub Action, and the Docker stack. Phases E–G are next.

| Phase | What it delivers |
|---|---|
| 1. Identity | Passport schema, SHA256 manifest, Merkle root, Ed25519 signing; `init`, `build`, `verify` |
| 2. Provenance | `passport run` records every stage's script, parameters, inputs, outputs, git state, and environment; optional MLflow and DVC |
| 3. Data checks | PII detection, k-anonymity, l-diversity, unique combinations, secrets scan, and the `policy.yaml` gate |
| 4. Model checks | Membership inference attack, train-test gap, pickle safety scan, dependency CVEs |
| 5. Sharing | Registry API, HTML report, JSON-LD export, Streamlit dashboard, Docker Compose |
| 6. Monitoring | Signed drift events, schema and live accuracy checks, linked and archived model versions |
| 7. Any data | `passport init --data --label`, automatic cleaning and typing, several model families chosen by cross-validation under an overfitting limit, regression support, `passport prepare` |
| B. Remediation | `passport init --llm` projects and `passport llm remediate`: sanitize, retrain, re-audit, and version until the gate passes, escalating from single values to whole types when needed. Every version is a linked, signed passport, with archived training data |
| A. Entity audit | `passport llm audit`, `sanitize`, `finetune`. The seven EL-MIA methods with the strongest chosen by cross-fitting; Gaussian null (LiRA); FDR control; likelihood × impact risk in CVSS bands; exposure-test confirmation; extraction probing for API models; audit results signed into the passport and its policy gate |

## Decisions

| Question | Decision | Why |
|---|---|---|
| Tabular or text? | Tabular first; Presidio is an optional extra for free text | Keeps the install light and CI fast |
| Dashboard | Streamlit | Fastest to build; the API stays UI-agnostic |
| Report rendering | Typed Python HTML builder, not Jinja2 | Type-checked, escape-by-default, one less dependency |
| Model families | Linear, gradient boosting, random forest (scikit-learn only) | No extra dependencies, and it handles missing values and categories natively |
| Model selection | Log loss / RMSE with k-fold CV, overfitting limit, one-standard-error rule | Proper scoring rules are less noisy than accuracy; the limit keeps the privacy gate passing |
| Missing evidence | `warn`, not `pass` | An unrun check is not a passed check |
| `unsafe_pickle` | `fail` | A pickle importing `os.system` or `eval` should never ship |

The passport also adds `artifacts`, `run`, and `revision` sections beyond the v0.1 spec.

## Platform build plan

MP is becoming an **entity-level PII leakage auditing and remediation platform**, following the product flow chart (2026-09-25). Why it exists:
- **Research gap:** LLMs memorize individual sensitive entities, and whole-document attacks miss this (EL-MIA, Satvaty et al., LREC 2026).
- **Market gap:** investors and acquirers now scrutinize where AI training data came from.

MP audits every sensitive entity, removes the risk, verifies the fix, and only then deploys. The passport engine built so far becomes its core: provenance (M9), the signed audit log (M4), and version history (M6).

Decisions: build all modules; audit open-weight and API models from the start; remediate by sanitizing data and retraining; set risk thresholds by industry practice. That means attack success at low false-positive rates (Carlini et al. 2022), CVSS-style severity bands, likelihood × impact risk (NIST SP 800-30), and findings mapped to OWASP LLM02 and NIST AI 100-2.

| Phase | Delivers | Modules |
|---|---|---|
| A. Entity audit engine (**done**) | Find sensitive entities in training text. Score each one's memorization with the EL-MIA methods (loss, zlib, min-k%, ReCaLL, and the reference-set attacks). Report per-entity risk with false-discovery control and AUC / TPR at low FPR. Works with Hugging Face models, OpenAI-compatible servers with log-probs, and extraction probing for API-only models. | M5 |
| B. Remediation (**done**) | Sanitize risky entities (surrogate, mask, or drop), fine-tune again from the base checkpoint, re-audit, and link the new version to the old one in a signed passport | M6, M7 |
| C. Backend (**done**) | FastAPI with PostgreSQL: tenants, 7 roles, RBAC, append-only audit log, per-tenant encrypted object storage, job queue and Python workers, provenance and diligence reports | M2, M3, M4, M9 |
| D. Control plane (**done**) | Go control plane (orchestration, deploy, rollback, kill switch) over gRPC, API gateway (subdomain → tenant, login check), and the `mp` CLI | M7, M8 |
| E. Web app | Next.js super-admin console and tenant dashboard for every module | M1–M9 |
| F. Release flow | Canary release to developer endpoints, verification (re-audit, canary testers, Claude/GPT probing), approval, consumer release, monitoring, rollback, kill switch | M8 |
| G. Infrastructure | Dockerfiles for every service, Docker Compose, Helm chart for Kubernetes, and Terraform for the cloud (network, cluster, database, encrypted storage) | |

## Phase C: what the backend includes

Everything is in `model_passport.platform`, tested end to end over HTTP on SQLite and on PostgreSQL with row-level security:
- **Organizations and access:** organizations, the seven roles, and a permission check on every request. The organization comes from the subdomain or an `X-MP-Tenant` header.
- **Data protection:** PostgreSQL row-level security, per-organization encryption of every file, and an append-only, hash-chained audit log.
- **Lifecycle:** a role-checked state machine: audit, findings, remediation, canary, verification, approval, release, and the kill switch.
- **Workers:** train, audit (with the release gate and a signed attestation), remediate (sanitize, retrain, next version), and report.
- **Diligence report:** lineage, provenance and consent, audit results, and approvals, with every signature and the audit chain re-verified.
- **API:** routes for M1–M4, M5–M7, and M9, plus the super-admin console.
- **CLI:** `passport platform keygen | migrate | create-superadmin | serve | worker`.

## Phase D: the control plane

`controlplane/` (Go):
- **`mp-controlplane`:** a gRPC service for deploy, rollback, the kill switch, listing deployments, and resolving which version serves a model. It uses the backend's roles and lifecycle rules and PostgreSQL row-level security, and appends to the same hash-chained audit log. The event hash matches Python byte for byte.
- **`mp-gateway`:** subdomain → organization. It strips any client-supplied organization header, checks the login at the edge, and blocks killed models.
- **`mp`:** the CLI for login, status, deploy, rollback, and kill.

The backend calls the control plane for canary, release, rollback, and kill. Integration tests build the Go binary, run it against PostgreSQL, and verify the shared audit chain from Python.

## Ideas for later

**Model quality**
- Optional LightGBM, XGBoost, or CatBoost families, and a stacked ensemble of the best candidates.
- Time-budgeted search (Optuna) instead of fixed grids.
- Nested cross-validation for an unbiased estimate of the selected model.
- Class weights for rare outcomes; group-aware splits when one person has many rows.
- Text columns as features (TF-IDF) instead of dropping them.
- Prediction intervals (conformal prediction) for regression.

**Privacy and security**
- Shadow-model membership inference (ML Privacy Meter or IBM ART) and attribute inference audits.
- Differential privacy training (Opacus or diffprivlib).
- Memorization tests for text models (canaries, PII extraction prompts).
- Key rotation, a trusted-key registry with revocation, and keyless signing with Sigstore.
- SLSA or in-toto provenance attestations alongside the passport.

**User experience**
- `passport doctor`: checks the setup (Python, key, config, data) and says what to fix.
- `passport explain`: the plain-language summary in the terminal.
- Dashboard: upload a CSV to train, compare two versions side by side, and export a model card (Markdown or PDF).
- Progress bars and time estimates during `passport run`.
- Slack or email alerts when drift is detected; scheduled monitoring in CI.

**Platform**
- Postgres backend for the registry, with pagination and single sign-on.
- Sync with the MLflow model registry and Hugging Face model cards.
- Airflow and Kubeflow operators.
- Regenerate the README screenshots automatically in CI.

## Developer notes

- Use `.venv` (Python 3.11 from uv at `~/.local/bin`): `uv pip install --python .venv/bin/python -e ".[dev,demo,registry,dashboard,mlflow,docs]"`, then `python -m playwright install chromium` for screenshots.
- Tests stub pip-audit (see `tests/conftest.py`); mark a test `@pytest.mark.network` to use the real one.
- Docker isn't installed on the dev machine, so the compose stack is checked only in CI. On failure the docker job publishes container logs as annotations, readable through the public API.
- Set `MLFLOW_DISABLE_AGENT_HINT=1` to silence MLflow's startup message.
