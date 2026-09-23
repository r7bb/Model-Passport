from __future__ import annotations

import hashlib
import secrets as pysecrets
import string
from pathlib import Path

from model_passport.scanners.base import ScanTarget
from model_passport.scanners.secrets import (
    SecretsScanner,
    confirmed_secrets,
    mask_secret,
    shannon_entropy,
)


def _random(alphabet: str, n: int) -> str:
    return "".join(pysecrets.choice(alphabet) for _ in range(n))


ALNUM = string.ascii_letters + string.digits


def _scan(tmp_path: Path, text: str) -> dict[str, object]:
    path = tmp_path / "file.py"
    path.write_text(text)
    return {f.category: f for f in SecretsScanner().scan(ScanTarget(path, name="file.py"))}


def test_entropy() -> None:
    assert shannon_entropy("aaaa") == 0
    assert shannon_entropy("abcd") == 2
    assert shannon_entropy(hashlib.sha256(b"x").hexdigest()) <= 4.0


def test_detects_known_patterns(tmp_path: Path) -> None:
    aws = "AKIA" + _random(string.ascii_uppercase + string.digits, 16)
    gh = "ghp_" + _random(ALNUM, 36)
    text = "\n".join(
        [
            f"AWS_KEY = '{aws}'",
            f"token: {gh}",
            "-----BEGIN OPENSSH PRIVATE KEY-----",
            f"password = '{_random(ALNUM, 16)}'",
            "sk-" + _random(ALNUM, 40),
        ]
    )
    hits = _scan(tmp_path, text)
    assert {"AWS_ACCESS_KEY_ID", "GITHUB_TOKEN", "PRIVATE_KEY", "GENERIC_SECRET_ASSIGNMENT",
            "LLM_API_KEY"} <= set(hits)  # fmt: skip
    assert hits["AWS_ACCESS_KEY_ID"].details["first_line"] == 1
    for finding in hits.values():
        for example in finding.masked_examples:
            assert example not in text  # masked, never raw
    assert aws not in str([f.model_dump() for f in hits.values()])


def test_ignores_benign_text(tmp_path: Path) -> None:
    text = "\n".join(
        [
            "password = input('Password: ')",
            "sha = '" + hashlib.sha256(b"x").hexdigest() + "'",
            "def compute_generalization_gap_between_train_and_test(): pass",
            "api_key = os.environ['API_KEY']",
        ]
    )
    hits = _scan(tmp_path, text)
    assert confirmed_secrets(list(hits.values())) == []


def test_high_entropy_is_medium_only(tmp_path: Path) -> None:
    hits = _scan(tmp_path, f"blob = '{_random(ALNUM + '+/', 48)}1a'")
    assert hits["HIGH_ENTROPY_STRING"].severity.value == "medium"
    assert confirmed_secrets(list(hits.values())) == []


def test_skips_binary(tmp_path: Path) -> None:
    path = tmp_path / "model.bin"
    path.write_bytes(b"\x00\x01AKIA" + b"A" * 16)
    assert SecretsScanner().scan(ScanTarget(path)) == []


def test_mask_secret() -> None:
    assert mask_secret("AKIAABCDEFGHIJKLMNOP") == "AKIA************"
