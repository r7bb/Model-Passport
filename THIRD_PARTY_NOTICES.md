# Third-party notices

Model Passport is released under the MIT License (see [LICENSE](LICENSE)). It builds on the work and software below, which remain under their own licenses.

## Research

- Kalokyri, V. et al. "AI Model Passport: Data and System Traceability Framework for Transparent AI in Health." arXiv:2506.22358 (2025), *Computational and Structural Biotechnology Journal*. Model Passport extends the AIPassport concept with automated privacy scanning, leakage auditing, and cryptographic identity. No code from AIPassport is included.
- Yeom, S. et al. "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting." IEEE CSF (2018). Basis of the loss-threshold membership inference audit.

## Runtime dependencies

| Package | License |
|---|---|
| cryptography | Apache-2.0 or BSD-3-Clause |
| pydantic | MIT |
| typer | MIT |
| PyYAML | MIT |
| pandas | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| SciPy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| PyArrow | Apache-2.0 |
| picklescan | MIT |
| joblib | BSD-3-Clause |
| pip-audit | Apache-2.0 |

## Optional dependencies

| Package | License | Used for |
|---|---|---|
| FastAPI, Uvicorn | MIT, BSD-3-Clause | Registry API |
| Streamlit | Apache-2.0 | Dashboard |
| MLflow | Apache-2.0 | Experiment tracking |
| DVC | Apache-2.0 | Data versioning |
| Presidio | MIT | Free-text PII detection |
| Faker | MIT | Synthetic demo data |

Vulnerability severities are retrieved from the [OSV](https://osv.dev) database (CC-BY-4.0 data, GitHub Advisory Database records).
