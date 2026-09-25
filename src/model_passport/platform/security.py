"""Passwords, session tokens, and per-tenant envelope encryption.

- Passwords are hashed with Argon2id (the OWASP-recommended default).
- Sessions are short-lived HS256 JWTs carrying the user id and a super-admin flag; tenant
  access is always re-checked against the database, never trusted from the token.
- Each tenant has its own random 256-bit data key, stored only wrapped (AES-256-GCM) by the
  platform master key. Objects are encrypted with the tenant key, so one tenant's key cannot
  read another's data, and rotating the master key only re-wraps small keys.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

NONCE_BYTES = 12
KEY_BYTES = 32
TOKEN_ALGORITHM = "HS256"  # noqa: S105 - an algorithm name, not a secret
_hasher = PasswordHasher()


class TokenError(ValueError):
    """The session token is missing, expired, or forged."""


class DecryptionError(ValueError):
    """Ciphertext does not decrypt under this key (wrong tenant, or tampered)."""


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return bool(_hasher.verify(password_hash, password))
    except (VerificationError, InvalidHashError):
        return False


def issue_token(user_id: str, super_admin: bool, secret: str, minutes: int) -> str:
    issued = datetime.now(UTC)
    claims = {
        "sub": user_id,
        "sa": super_admin,
        "iat": issued,
        "exp": issued + timedelta(minutes=minutes),
    }
    return jwt.encode(claims, secret, algorithm=TOKEN_ALGORITHM)


def read_token(token: str, secret: str) -> dict[str, Any]:
    try:
        claims: dict[str, Any] = jwt.decode(
            token, secret, algorithms=[TOKEN_ALGORITHM], options={"require": ["sub", "exp"]}
        )
    except jwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc
    return claims


def encrypt(key: bytes, plaintext: bytes, context: bytes = b"") -> bytes:
    """AES-256-GCM; ``context`` (e.g. tenant id + object key) is authenticated, not stored."""
    nonce = os.urandom(NONCE_BYTES)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, context)


def decrypt(key: bytes, blob: bytes, context: bytes = b"") -> bytes:
    try:
        return AESGCM(key).decrypt(blob[:NONCE_BYTES], blob[NONCE_BYTES:], context)
    except Exception as exc:  # cryptography raises InvalidTag
        raise DecryptionError("cannot decrypt: wrong key or tampered data") from exc


def new_data_key() -> bytes:
    return os.urandom(KEY_BYTES)


def wrap_key(master: bytes, data_key: bytes, tenant_id: str) -> bytes:
    return encrypt(master, data_key, b"tenant-key:" + tenant_id.encode())


def unwrap_key(master: bytes, wrapped: bytes, tenant_id: str) -> bytes:
    return decrypt(master, wrapped, b"tenant-key:" + tenant_id.encode())
