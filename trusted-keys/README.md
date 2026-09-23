# Trusted signing keys

Public keys (`*.pub`) placed here are mounted into the registry container. `GET /passports/{id}/verify` then reports `key_trusted: true` for passports signed by one of them.

Only public keys belong here. Private keys (`*.pem`) are gitignored and must never be committed.
