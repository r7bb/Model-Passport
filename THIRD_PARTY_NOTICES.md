# Third-party notices

Model Passport is released under the MIT License (see [LICENSE](LICENSE)). It builds on the work and software below, which remain under their own licenses.

## Research

- Kalokyri, V. et al. "AI Model Passport: Data and System Traceability Framework for Transparent AI in Health." arXiv:2506.22358 (2025), *Computational and Structural Biotechnology Journal*. Model Passport extends the AIPassport concept with automated privacy scanning, leakage auditing, and cryptographic identity. No code from AIPassport is included.
- Yeom, S. et al. "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting." IEEE CSF (2018). Basis of the loss-threshold membership inference audit.

## Datasets

| Dataset | License | Source | Attribution |
|---|---|---|---|
| Nemotron-PII (NVIDIA) | CC BY 4.0 | https://huggingface.co/datasets/nvidia/Nemotron-PII | "Nemotron-PII by NVIDIA, licensed under CC BY 4.0" |

Used by `model_passport.llm.public_data` for the public accuracy benchmark. The dataset is downloaded on demand; this repository includes only aggregate benchmark results computed from it (`docs/benchmarks/nemotron-pythia-160m.json`).

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

## Other optional and web dependencies (to be confirmed)

Copied as-is from the installed packages' own metadata (Python `License-Expression`, else `License`, else the license classifier; npm `license`). Not yet reviewed; confirm each against the package's license file before release.

| Package | Version installed | Metadata says |
|---|---|---|
| torch | 2.14.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| transformers | 5.17.0 | Apache 2.0 License |
| tokenizers | 0.23.2 | Apache Software License (classifier) |
| safetensors | 0.8.0 | Apache Software License (classifier) |
| httpx | 0.28.1 | BSD-3-Clause |
| SQLAlchemy | 2.0.54 | MIT |
| alembic | 1.20.0 | MIT |
| psycopg | 3.3.6 | LGPL-3.0-only |
| PyJWT | 2.15.0 | MIT |
| argon2-cffi | 25.1.0 | MIT |
| boto3 | 1.43.103 | Apache-2.0 |
| python-multipart | 0.0.32 | Apache-2.0 |
| grpcio | 1.84.0 | Apache-2.0 |
| protobuf | 7.36.2 | 3-Clause BSD License |
| mlflow | 3.16.1 | "Copyright 2018 Databricks, Inc.  All rights reserved." (license field); Apache Software License (classifier) |
| presidio-analyzer | not installed | not checked |
| next | 16.3.6 | MIT |
| react | 19.2.8 | MIT |
| react-dom | 19.2.8 | MIT |
| server-only | 0.0.1 | MIT |

Vulnerability severities are retrieved from the [OSV](https://osv.dev) database (CC-BY-4.0 data, GitHub Advisory Database records).
