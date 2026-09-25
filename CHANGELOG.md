# Changelog

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
