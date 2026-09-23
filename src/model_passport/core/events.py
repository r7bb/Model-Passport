"""Append-only, hash-chained, individually signed lifecycle events.

Each event stores ``prev_event_sha256``: the hash of the previous event (with its signature),
or for the first event the hash of the passport's signed payload. Reordering, deleting, or
editing any event, or moving events to a different passport, breaks the chain.
"""

from __future__ import annotations

from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from model_passport.core import identity
from model_passport.core.schema import LifecycleEvent


def _event_body(event: dict[str, Any]) -> bytes:
    return identity.canonical_json({**event, "signature": None})


def anchor_hash(passport: dict[str, Any]) -> str:
    return identity.sha256_bytes(identity.signing_payload(passport))


def event_hash(event: dict[str, Any]) -> str:
    return identity.sha256_bytes(identity.canonical_json(event))


def next_prev_hash(passport: dict[str, Any]) -> str:
    events = passport.get("events") or []
    return event_hash(events[-1]) if events else anchor_hash(passport)


def append_event(
    passport: dict[str, Any],
    event_type: str,
    payload: dict[str, Any],
    private_key: Ed25519PrivateKey,
) -> dict[str, Any]:
    """Sign a new event, append it to ``passport["events"]`` in place, and return it."""
    event = LifecycleEvent(
        event_type=event_type, payload=payload, prev_event_sha256=next_prev_hash(passport)
    ).model_dump(mode="json")
    event["signature"] = identity.sign(private_key, _event_body(event))
    passport.setdefault("events", []).append(event)
    return event


def verify_events(passport: dict[str, Any], public_key: Ed25519PublicKey) -> list[str]:
    """Return a list of problems with the event chain (empty when valid)."""
    errors = []
    expected_prev = anchor_hash(passport)
    for index, event in enumerate(passport.get("events") or []):
        label = f"event {index} ({event.get('event_type', '?')})"
        if event.get("prev_event_sha256") != expected_prev:
            errors.append(f"{label}: chain broken (reordered, removed, or foreign event)")
        signature = event.get("signature")
        if not signature or not identity.verify_signature(
            public_key, _event_body(event), signature
        ):
            errors.append(f"{label}: invalid signature")
        expected_prev = event_hash(event)
    return errors
