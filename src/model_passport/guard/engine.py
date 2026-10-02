"""The privacy layer between an app and an AI model: protect what goes in, review what comes out.

**Requests.** Personal values in a request are replaced before it reaches the model. With the
``mask`` action a value becomes a placeholder such as ``[EMAIL_1]``; the same value always gets
the same placeholder within a request, and the reply's placeholders are turned back into the
real values, so the user gets a complete answer while the model never saw the data. ``redact``
replaces irreversibly, ``block`` refuses the request, and ``allow`` lets the value through.

**Replies.** A personal value in a reply that the user did not send came from the model: from
its training data, its context, or a tool. High-impact values are redacted by default, and a
value the entity audit found memorized (matched by keyed fingerprint, so the gateway never
stores the values themselves) gets the ``memorized`` action.

Nothing here stores a raw value. The vault lives for one request.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from model_passport.llm.entities import DETECTED_TYPES, HIGH, Span, detect, impact


class Action(StrEnum):
    MASK = "mask"  # reversible placeholder, restored in the reply
    REDACT = "redact"  # irreversible placeholder
    BLOCK = "block"  # refuse the whole request or reply
    ALLOW = "allow"


class Detector(Protocol):
    def find(self, text: str) -> list[Span]: ...


class PatternDetector:
    """Emails, phone numbers, SSNs, card numbers, IP addresses, and dates, by pattern.

    Names and addresses need a model-based detector (roadmap H3); plug one in with the same
    ``find`` method.
    """

    types = DETECTED_TYPES

    def find(self, text: str) -> list[Span]:
        return detect(text)


# Dates pass by default: a pattern cannot tell a birth date from "the meeting on Oct 5", and
# masking every date would stop the model reasoning about schedules. Policies can change it.
REQUEST_DEFAULTS = {"DATE": Action.ALLOW}


@dataclass(frozen=True)
class Policy:
    """What to do with each type of value, in requests and in replies."""

    inbound: Mapping[str, Action] = field(default_factory=dict)
    inbound_default: Action = Action.MASK
    outbound: Mapping[str, Action] = field(default_factory=dict)
    memorized: Action = Action.REDACT

    def for_request(self, entity_type: str) -> Action:
        default = REQUEST_DEFAULTS.get(entity_type, self.inbound_default)
        return self.inbound.get(entity_type, default)

    def for_reply(self, entity_type: str) -> Action:
        """High-impact values (SSNs, cards, account numbers, ...) are redacted unless set."""
        default = Action.REDACT if impact(entity_type) >= HIGH else Action.ALLOW
        return self.outbound.get(entity_type, default)

    @classmethod
    def from_json(cls, data: Mapping[str, object] | None) -> Policy:
        data = data or {}

        def actions(key: str) -> dict[str, Action]:
            raw = data.get(key) or {}
            if not isinstance(raw, Mapping):
                raise TypeError(f"{key} must map entity types to actions")
            return {str(k).upper(): Action(str(v)) for k, v in raw.items()}

        return cls(
            inbound=actions("inbound"),
            inbound_default=Action(str(data.get("inbound_default", Action.MASK))),
            outbound=actions("outbound"),
            memorized=Action(str(data.get("memorized", Action.REDACT))),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "inbound": {k: v.value for k, v in self.inbound.items()},
            "inbound_default": self.inbound_default.value,
            "outbound": {k: v.value for k, v in self.outbound.items()},
            "memorized": self.memorized.value,
        }


PLACEHOLDER = re.compile(r"\[([A-Z][A-Z0-9]*)_(\d+)\]")


class Vault:
    """Placeholders for one request: the same value always gets the same placeholder."""

    def __init__(self) -> None:
        self._token: dict[tuple[str, str], str] = {}
        self._value: dict[str, str] = {}
        self._next: Counter[str] = Counter()

    def placeholder(self, entity_type: str, value: str) -> str:
        key = (entity_type, value)
        if key not in self._token:
            self._next[entity_type] += 1
            token = f"[{entity_type}_{self._next[entity_type]}]"
            self._token[key] = token
            self._value[token] = value
        return self._token[key]

    def knows(self, value: str) -> bool:
        return any(v == value for v in self._value.values())

    def restore(self, text: str, escape: Callable[[str], str] | None = None) -> str:
        """Placeholders back to values; ``escape`` encodes them, e.g. inside JSON strings."""

        def value(match: re.Match[str]) -> str:
            original = self._value.get(match.group(0))
            if original is None:
                return match.group(0)
            return escape(original) if escape else original

        return PLACEHOLDER.sub(value, text)

    def __len__(self) -> int:
        return len(self._value)


class BlockedError(Exception):
    """The policy refuses this request or reply; ``types`` says which kinds of value."""

    def __init__(self, where: str, types: list[str]) -> None:
        super().__init__(f"{where} blocked: contains {', '.join(sorted(set(types)))}")
        self.where = where
        self.types = sorted(set(types))


@dataclass
class Report:
    """What the guard did, by entity type (never the values)."""

    masked: Counter[str] = field(default_factory=Counter)
    redacted: Counter[str] = field(default_factory=Counter)
    leaked: Counter[str] = field(default_factory=Counter)  # values the model produced
    memorized: Counter[str] = field(default_factory=Counter)

    def to_json(self) -> dict[str, dict[str, int]]:
        return {
            "masked": dict(self.masked),
            "redacted": dict(self.redacted),
            "leaked": dict(self.leaked),
            "memorized": dict(self.memorized),
        }


def fingerprint(key: bytes, entity_type: str, value: str) -> str:
    """The keyed hash the entity audit stores for a finding (see ``llm.audit``)."""
    return hmac.new(key, f"{entity_type}:{value}".encode(), hashlib.sha256).hexdigest()


Memorized = Callable[[str, str], bool]


def _never(_: str, __: str) -> bool:
    return False


@dataclass
class Guard:
    policy: Policy = field(default_factory=Policy)
    detector: Detector = field(default_factory=PatternDetector)
    is_memorized: Memorized = _never

    def protect(self, text: str, vault: Vault, report: Report) -> str:
        """The request text as the model may see it."""
        spans = self.detector.find(text)
        blocked = [
            s.entity_type for s in spans if self.policy.for_request(s.entity_type) is Action.BLOCK
        ]
        if blocked:
            raise BlockedError("request", blocked)
        out, last = [], 0
        for span in spans:
            value = text[span.start : span.end]
            action = self.policy.for_request(span.entity_type)
            out.append(text[last : span.start])
            if action is Action.MASK:
                out.append(vault.placeholder(span.entity_type, value))
                report.masked[span.entity_type] += 1
            elif action is Action.REDACT:
                out.append(f"[{span.entity_type}]")
                report.redacted[span.entity_type] += 1
            else:
                out.append(value)
            last = span.end
        out.append(text[last:])
        return "".join(out)

    def review(
        self,
        text: str,
        vault: Vault,
        report: Report,
        escape: Callable[[str], str] | None = None,
    ) -> str:
        """The model's reply as the user may see it, with placeholders restored."""
        spans = [s for s in self.detector.find(text) if not vault.knows(text[s.start : s.end])]
        decisions = []
        for span in spans:
            value = text[span.start : span.end]
            report.leaked[span.entity_type] += 1
            if self.is_memorized(span.entity_type, value):
                report.memorized[span.entity_type] += 1
                decisions.append((span, self.policy.memorized))
            else:
                decisions.append((span, self.policy.for_reply(span.entity_type)))
        blocked = [s.entity_type for s, action in decisions if action is Action.BLOCK]
        if blocked:
            raise BlockedError("reply", blocked)
        out, last = [], 0
        for span, action in decisions:
            out.append(text[last : span.start])
            if action in (Action.REDACT, Action.MASK):
                out.append(f"[{span.entity_type}]")
                report.redacted[span.entity_type] += 1
            else:
                out.append(text[span.start : span.end])
            last = span.end
        out.append(text[last:])
        return vault.restore("".join(out), escape)
