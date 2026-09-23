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

### Phase 2: Provenance capture
- `stages:` in `passport.yaml`; `passport run` executes each stage and records command, script hash, git commit and dirty state, parameters, input and output hashes, and timings.
- Parameter overrides with `passport run --set stage.key=value`, passed to scripts through `PASSPORT_PARAMS`.
- Environment capture: Python, OS, hardware, installed packages.
- Optional MLflow logging (one run per stage, passport attached at build) and DVC tracking of stage outputs.

### Phase 3: Data scanners and policy
- Common `Scanner` interface returning `Finding` objects (counts, categories, masked examples only).
- PII scanner: column name heuristics plus validators for email, phone, SSN, Luhn-checked credit cards, IP addresses, and dates of birth. Sampling with an optional full scan.
- Reidentification risk: k anonymity, l diversity, unique record fraction, and the smallest column combinations that make records unique.
- Secrets scanner: known token patterns plus Shannon entropy, over data files and pipeline scripts.
- Policy engine driven by `policy.yaml`, with pass, warn, and fail per rule. `passport build` exits nonzero on fail.

### Phase 4: Model auditors
- Loss threshold membership inference: attack AUC and TPR at 1% FPR.
- Generalization gap between train and test metrics.
- Pickle opcode scan (picklescan); warns when raw pickle is used instead of safetensors or ONNX.
- `pip-audit` of the captured environment, with CVE severity from OSV.

### Phase 5: Registry, report, dashboard
- FastAPI and SQLite registry: upload, list, get, verify, and events endpoints, with an optional bearer token for writes.
- Static `passport.html` report rendered with Jinja2, and JSON-LD export using W3C PROV terms.
- Streamlit dashboard: model picker, verdict, lineage, pipeline DAG, scan results, and a plain-language "simple view".
- Docker Compose running MLflow, MinIO (as MLflow's artifact store), the registry, and the dashboard.

### Phase 6: CI and monitoring
- GitHub Actions: lint, tests, and an end-to-end demo run through `passport build` and `passport verify`. CI fails if the demo data carries PII.
- Drift checks (Kolmogorov-Smirnov, Mann-Whitney U, and Cramér-von Mises for numeric features; chi-square for categorical) appended as hash-chained, signed lifecycle events.

## Later
- Shadow model membership inference via ML Privacy Meter or IBM ART.
- Text model memorization: canary exposure tests and PII extraction prompts.
- Differential privacy training (Opacus) in the demo.
- Key rotation, and a trusted key registry with revocation.
- Postgres backend for the registry; React dashboard.
- Signed model cards exported to Hugging Face and MLflow model registry tags.
- DCAT and ML Schema alignment alongside PROV in the JSON-LD export.
