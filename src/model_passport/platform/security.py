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
import threading
import time
from collections import OrderedDict, defaultdict, deque
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


class LoginThrottle:
    """Limit failed sign-ins per account and per client address (sliding window).

    Stops password guessing: after ``per_account`` failures for one email, or ``per_client``
    from one address, within ``window`` seconds, further attempts are refused until the oldest
    failure ages out. A successful sign-in clears that account's failures. State is per
    process; behind several replicas, pair it with a limit at the gateway or load balancer.

    The per-address limit is off by default (``per_client=0``): behind the web app or a proxy
    every sign-in arrives from the same address, and one attacker could lock everyone out.
    """

    def __init__(self, per_account: int = 5, per_client: int = 0, window: float = 900.0) -> None:
        self.limits = {"account": per_account, "client": per_client}
        self.window = window
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _keys(self, email: str, client: str) -> dict[str, str]:
        return {"account": f"account:{email}", "client": f"client:{client}"}

    def retry_after(self, email: str, client: str, now: float | None = None) -> int:
        """Seconds until another attempt is allowed (0 if allowed now)."""
        now = time.monotonic() if now is None else now
        wait = 0.0
        with self._lock:
            for kind, key in self._keys(email, client).items():
                if self.limits[kind] <= 0:
                    continue
                failures = self._failures[key]
                while failures and failures[0] <= now - self.window:
                    failures.popleft()
                if len(failures) >= self.limits[kind]:
                    wait = max(wait, failures[0] + self.window - now)
        return int(wait) + 1 if wait > 0 else 0

    def failed(self, email: str, client: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            for key in self._keys(email, client).values():
                self._failures[key].append(now)

    def succeeded(self, email: str) -> None:
        with self._lock:
            self._failures.pop(f"account:{email}", None)


class RateLimiter:
    """Allow ``limit`` requests per ``window`` seconds for each key (fixed window).

    ``check`` counts a request and returns 0 if it may go ahead, or the seconds to wait. Memory
    is bounded: expired windows are dropped when the table fills, and if it is still full the
    key idle the longest is forgotten, so at most ``max_keys`` keys are tracked. A ``limit`` of
    0 turns it off. State is in memory and per process; behind several replicas each one
    counts separately, so pair it with a limit at the gateway or load balancer.
    """

    def __init__(self, limit: int, window: float = 60.0, max_keys: int = 10_000) -> None:
        self.limit, self.window, self.max_keys = limit, window, max_keys
        self._windows: OrderedDict[str, tuple[float, int]] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._windows)

    def check(self, key: str, now: float | None = None) -> int:
        if self.limit <= 0:
            return 0
        now = time.monotonic() if now is None else now
        with self._lock:
            started, count = self._windows.get(key, (now, 0))
            if now - started >= self.window:
                started, count = now, 0
            if key not in self._windows:
                self._make_room(now)
            self._windows[key] = (started, count + 1)
            self._windows.move_to_end(key)
            if count + 1 <= self.limit:
                return 0
            return int(started + self.window - now) + 1

    def _make_room(self, now: float) -> None:
        if len(self._windows) < self.max_keys:
            return
        expired = [k for k, (start, _) in self._windows.items() if now - start >= self.window]
        for key in expired:
            del self._windows[key]
        while len(self._windows) >= self.max_keys:
            self._windows.popitem(last=False)
