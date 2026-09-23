from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from model_passport.core import identity


def test_sha256_file_streams_large_file(tmp_path: Path) -> None:
    data = os.urandom(identity.CHUNK_SIZE * 3 + 17)
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert identity.sha256_file(path) == hashlib.sha256(data).hexdigest()


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def test_merkle_root_is_order_independent() -> None:
    hashes = [_h(str(i)) for i in range(5)]
    assert identity.merkle_root(hashes) == identity.merkle_root(reversed(hashes))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8])
def test_merkle_root_changes_when_any_leaf_changes(n: int) -> None:
    hashes = [_h(str(i)) for i in range(n)]
    root = identity.merkle_root(hashes)
    for i in range(n):
        altered = hashes.copy()
        altered[i] = _h(f"tampered-{i}")
        assert identity.merkle_root(altered) != root


def test_merkle_root_single_leaf_is_domain_separated() -> None:
    h = _h("only")
    assert identity.merkle_root([h]) == hashlib.sha256(b"\x00" + bytes.fromhex(h)).hexdigest()
    assert identity.merkle_root([h]) != h


def test_merkle_root_odd_leaf_not_duplicated() -> None:
    a, b, c = sorted(_h(x) for x in "abc")
    assert identity.merkle_root([a, b, c]) != identity.merkle_root([a, b, c, c])


def test_merkle_root_rejects_empty() -> None:
    with pytest.raises(ValueError):
        identity.merkle_root([])


def test_canonical_json_is_deterministic() -> None:
    assert identity.canonical_json({"b": 1, "a": [1, {"d": 2, "c": 3}]}) == (
        b'{"a":[1,{"c":3,"d":2}],"b":1}'
    )


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        identity.canonical_json({"x": float("nan")})


def test_signing_payload_ignores_signature_and_events() -> None:
    base = {"identity": {"signature": None, "model_name": "m"}, "events": []}
    signed = {"identity": {"signature": "abc", "model_name": "m"}, "events": [{"e": 1}]}
    assert identity.signing_payload(base) == identity.signing_payload(signed)
    assert signed["identity"]["signature"] == "abc"  # input not mutated


def test_sign_and_verify_roundtrip() -> None:
    key = identity.generate_keypair()
    sig = identity.sign(key, b"payload")
    assert identity.verify_signature(key.public_key(), b"payload", sig)
    assert not identity.verify_signature(key.public_key(), b"payload!", sig)
    assert not identity.verify_signature(identity.generate_keypair().public_key(), b"payload", sig)
    assert not identity.verify_signature(key.public_key(), b"payload", "not base64!!")


def test_keypair_save_load(tmp_path: Path) -> None:
    key = identity.generate_keypair()
    priv, pub = tmp_path / "k" / "key.pem", tmp_path / "k" / "key.pub"
    identity.save_keypair(key, priv, pub)

    assert stat.S_IMODE(priv.stat().st_mode) == 0o600
    loaded = identity.load_private_key(priv)
    assert identity.public_key_fingerprint(loaded.public_key()) == (
        identity.public_key_fingerprint(identity.load_public_key(pub))
    )
    assert identity.public_key_fingerprint(key.public_key()).startswith("sha256:")


def test_keypair_passphrase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(identity.PASSPHRASE_ENV, "correct horse")
    priv, pub = tmp_path / "key.pem", tmp_path / "key.pub"
    identity.save_keypair(identity.generate_keypair(), priv, pub)
    assert b"ENCRYPTED" in priv.read_bytes()
    identity.load_private_key(priv)

    monkeypatch.setenv(identity.PASSPHRASE_ENV, "wrong")
    with pytest.raises(ValueError):
        identity.load_private_key(priv)
