"""Cryptographic identity: SHA256 hashing, Merkle root, canonical JSON, Ed25519 signing."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

CHUNK_SIZE = 1 << 20  # 1 MiB
PASSPHRASE_ENV = "PASSPORT_KEY_PASSPHRASE"  # noqa: S105 - env var name

# Domain separation prefixes (RFC 6962 style) so a leaf can never be confused with a node.
_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"


# --- Hashing -----------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Stream a file through SHA256 and return the lowercase hex digest."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def merkle_root(hashes: Iterable[str]) -> str:
    """Merkle root over the sorted list of hex SHA256 artifact hashes.

    Leaves are ``H(0x00 || hash)``, internal nodes ``H(0x01 || left || right)``. An odd node
    at the end of a level is promoted unchanged rather than duplicated, which avoids the
    duplicate-leaf ambiguity of Bitcoin-style trees.
    """
    leaves = sorted(hashes)
    if not leaves:
        raise ValueError("Merkle root requires at least one artifact hash")
    level = [hashlib.sha256(_LEAF_PREFIX + bytes.fromhex(h)).digest() for h in leaves]
    while len(level) > 1:
        paired = [
            hashlib.sha256(_NODE_PREFIX + level[i] + level[i + 1]).digest()
            for i in range(0, len(level) - 1, 2)
        ]
        if len(level) % 2:
            paired.append(level[-1])
        level = paired
    return level[0].hex()


# --- Canonical JSON ----------------------------------------------------------------------


def canonical_json(data: Any) -> bytes:
    """Deterministic JSON encoding: sorted keys, no whitespace, UTF-8."""
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def signing_payload(passport: dict[str, Any]) -> bytes:
    """Bytes covered by the passport signature.

    The signature field itself is blanked, and ``events`` is excluded because lifecycle
    events are appended after signing and carry their own signatures.
    """
    body = copy.deepcopy(passport)
    body.get("identity", {})["signature"] = None
    body.pop("events", None)
    return canonical_json(body)


class KeyFormatError(ValueError):
    """Raised when a key file holds something other than an Ed25519 key."""


# --- Keys --------------------------------------------------------------------------------


def generate_keypair() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def _passphrase() -> bytes | None:
    value = os.environ.get(PASSPHRASE_ENV)
    return value.encode("utf-8") if value else None


def save_keypair(private_key: Ed25519PrivateKey, private_path: Path, public_path: Path) -> None:
    """Write the private key (mode 0600, encrypted if a passphrase is set) and public key."""
    passphrase = _passphrase()
    encryption: serialization.KeySerializationEncryption = (
        serialization.BestAvailableEncryption(passphrase)
        if passphrase
        else serialization.NoEncryption()
    )
    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, encryption
    )
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(private_pem)
    public_path.write_bytes(public_pem)


def write_public_key(private_key: Ed25519PrivateKey, public_path: Path) -> None:
    """Write the public half of ``private_key`` (e.g. when only the private key was supplied)."""
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )


def load_private_key(path: Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=_passphrase())
    if not isinstance(key, Ed25519PrivateKey):
        raise KeyFormatError(f"{path} is not an Ed25519 private key")
    return key


def load_public_key(path: Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(path.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise KeyFormatError(f"{path} is not an Ed25519 public key")
    return key


def public_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return "sha256:" + sha256_bytes(raw)


# --- Signing -----------------------------------------------------------------------------


def sign(private_key: Ed25519PrivateKey, payload: bytes) -> str:
    return base64.b64encode(private_key.sign(payload)).decode("ascii")


def verify_signature(public_key: Ed25519PublicKey, payload: bytes, signature: str) -> bool:
    try:
        public_key.verify(base64.b64decode(signature, validate=True), payload)
    except (InvalidSignature, ValueError):
        return False
    return True
