# Model Passport roadmap

Model Passport extends the AIPassport framework (Kalokyri et al., arXiv 2506.22358) with privacy scanning, leakage auditing, artifact safety checks, signed identity, a CI policy gate, monitoring, and adaptive training on any tabular data.

## Where things stand

All seven phases are done: 198 tests at 92% coverage, strict ruff and mypy, and green CI (lint, tests on Python 3.11 and 3.12, the full demo lifecycle, and the Docker stack).

| Phase | What it delivers |
|---|---|
| 1. Identity | Passport schema, SHA256 manifest, Merkle root, Ed25519 signing; `init`, `build`, `verify` |
| 2. Provenance | `passport run` records every stage's script, parameters, inputs, outputs, git state, and environment; optional MLflow and DVC |
| 3. Data checks | PII detection, k-anonymity, l-diversity, unique combinations, secrets scan, and the `policy.yaml` gate |
| 4. Model checks | Membership inference attack, train-test gap, pickle safety scan, dependency CVEs |
| 5. Sharing | Registry API, HTML report, JSON-LD export, Streamlit dashboard, Docker Compose |
| 6. Monitoring | Signed drift events, schema and live accuracy checks, linked and archived model versions |
| 7. Any data | `passport init --data --label`, automatic cleaning and typing, several model families chosen by cross-validation under an overfitting limit, regression support, `passport prepare` |

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

## Next up

In priority order:

1. **Easy install (built, waiting to publish):** the package, GitHub Action, and release workflow for PyPI and GHCR are ready and tested in CI. The first release needs the one-time PyPI setup in SETUP.md, then a `v0.2.0` tag.
2. **Safer model files:** save models with skops or ONNX instead of pickle, which removes the "raw pickle" warning.
3. **Fairness checks:** accuracy and error rates per group (for example, by gender or age range), with policy limits.
4. **Explanations:** the features that matter most (permutation importance) shown in the report and dashboard.
5. **Better probabilities:** calibration and a decision threshold tuned for the goal (accuracy, recall, or cost).
6. **Time-aware splits:** when data has a date column, train on the past and test on the future, so the score reflects real use.

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
