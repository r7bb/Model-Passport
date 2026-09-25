# Changelog

## Unreleased

MP becomes an entity-level PII leakage auditing and remediation platform (phase A of the platform build plan).

- `passport llm audit`: an entity-level membership inference audit (EL-MIA) of a language model on its training corpus. It runs seven methods and uses the strongest, chosen by cross-fitting. Each entity gets a p-value against controls (robust Gaussian null), false-discovery control, and a likelihood × impact risk in CVSS bands. High and Critical findings are confirmed with the exposure test. Reports include AUC and TPR at low FPR.
- Backends: Hugging Face models, OpenAI-compatible servers with prompt log-probs (likelihood attacks), and Anthropic and OpenAI chat models (extraction probing).
- `passport llm sanitize` (surrogate, mask, or drop) and `passport llm finetune` (safetensors) for the remediation loop; `passport llm demo-corpus` for synthetic data.
- `passport init --llm` and `passport llm remediate` (phase B): a project that tests, sanitizes, retrains, and re-tests as new signed versions (1.0.0 → 1.1.0 → ...) until the release gate passes. Remediation escalates from single values to whole types when a type keeps leaking.
- The release gate for language models blocks confirmed High and Critical findings.
- Passports can carry the entity audit (`privacy.entity_audit`), and policies gain `entity_critical_max`, `entity_high_max`, `el_mia_auc_max`, and `el_mia_tpr_at_1pct_fpr_max`.
- Model directories (Hugging Face checkpoints) are hashed as one artifact, verified file by file, and scanned for unsafe weights.
- Docker Compose uses SeaweedFS for S3-compatible storage (MinIO images are no longer freely available).

## 0.2.0

Works on any tabular dataset, and is ready to install without cloning the repository.

- `passport init --data <file> --label <column>` sets up ready-made stages for your own data.
- Adaptive training (`model_passport.ml`): column types, task (binary, multiclass, regression), and row cleaning are detected automatically. Linear, gradient-boosting, and random-forest models are compared by cross-validation, and candidates that overfit are rejected.
- Adaptive k-anonymity: quasi-identifiers are coarsened step by step, each step chosen by people protected per unit of information lost about the label (columns that don't predict the label go first). Train and test are split within each group, and at most 1% of records are removed.
- `passport prepare` prepares new batches with the certified preprocessing.
- `passport dashboard` opens the dashboard, which is now part of the package.
- `passport --version`, and the help is grouped by workflow step.
- The leakage audit and monitoring support regression models.
- Drift checks compare numbers stored as text as numbers and dates as dates, skip ID columns, and pool rare categories.
- A reusable GitHub Action (`uses: r7bb/Model-Passport@v0.2.0`) and a release workflow for PyPI and GHCR.

## 0.1.0

First release: signed passports with provenance capture, PII and reidentification scans, the membership inference audit, artifact and dependency checks, the policy gate, drift monitoring, the registry, the HTML report, the dashboard, and Docker Compose.
