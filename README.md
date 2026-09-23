# Model-Passport
Model Passport is an MLOps tool that wraps an ML training pipeline and produces a signed, verifiable "passport" for every trained model. The passport records where the model came from (data, code, parameters, environment, metrics) and certifies that the training data and the trained model were checked for privacy and security vulnerabilities.

## Status

Phases 1 and 2 are implemented: the signed passport (schema, SHA256 hashing, Merkle root, Ed25519 signing, `init`, `build`, `verify`) and provenance capture (`passport run`, with git, environment, optional MLflow and DVC). See [ROADMAP.md](ROADMAP.md).

## Setup

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev,demo]"
```

## Quickstart

```bash
passport init --name my-model     # writes passport.yaml and a signing keypair in .passport/
# edit passport.yaml: stages, model path, datasets, declared fields
passport run                      # runs stages, records scripts, params, inputs, outputs, git state
passport build                    # hashes artifacts, signs, writes passport.json
passport verify passport.json     # exits 1 and names any changed or missing file
```

Stage parameters are passed to scripts as JSON in `$PASSPORT_PARAMS`; read them with `model_passport.runtime.params(defaults)`. Override them per run with `passport run --set train.model=overfit`.

## Demo

```bash
python demo/make_dataset.py       # synthetic income data with injected fake PII (Faker)
passport init
passport run && passport build && passport verify
```

The private key in `.passport/` is gitignored and must never be committed. Set `PASSPORT_KEY_PASSPHRASE` to encrypt it at rest.

## How verification works

- Every artifact (config, model, datasets, scripts) is hashed with streamed SHA256 and listed in the passport's `artifacts` manifest.
- A Merkle root is computed over the sorted artifact hashes, with domain-separated leaf and node hashes.
- The passport is canonicalized (sorted keys, no whitespace) and signed with Ed25519. The `signature` field and the append-only `events` list are excluded from the signed payload; events are signed individually.
- `verify` re-hashes each file, rebuilds the Merkle root, checks the public key fingerprint, and checks the signature against the passport exactly as stored on disk.

## Development

```bash
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```
