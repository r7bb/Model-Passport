# Model-Passport
Model Passport is an MLOps tool that wraps an ML training pipeline and produces a signed, verifiable "passport" for every trained model. The passport records where the model came from (data, code, parameters, environment, metrics) and certifies that the training data and the trained model were checked for privacy and security vulnerabilities.

## Status

Phase 1 (core passport and identity) is implemented: schema, SHA256 artifact hashing, Merkle root, Ed25519 signing, and the `init`, `build`, and `verify` commands.

## Setup

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Quickstart

```bash
passport init --name my-model     # writes passport.yaml and a signing keypair in .passport/
# edit passport.yaml: model path, datasets, scripts, metrics, declared fields
passport build                    # hashes artifacts, signs, writes passport.json
passport verify passport.json     # exits 1 and names any changed or missing file
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
