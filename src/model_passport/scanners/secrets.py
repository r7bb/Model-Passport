"""Secrets scanner: known credential patterns plus Shannon entropy, over text files."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from model_passport.core.schema import Finding, Severity, at_least
from model_passport.scanners.base import Scanner, ScanTarget

BINARY_SNIFF_BYTES = 8192
MAX_FILE_BYTES = 200 * 1024 * 1024


@dataclass(frozen=True)
class SecretPattern:
    name: str
    regex: re.Pattern[str]
    severity: Severity
    group: int = 0  # capture group holding the secret value
    min_entropy: float = 0.0
    literal_only: bool = False  # skip values that look like code (os.environ, getenv(...), ids)


CODE_REFERENCE_RE = re.compile(r"[()\[\]{}<>$]|^[A-Za-z_][A-Za-z_.]*$")


PATTERNS = [
    SecretPattern("PRIVATE_KEY", re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----"
    ), Severity.CRITICAL),
    SecretPattern("AWS_ACCESS_KEY_ID", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
                  Severity.CRITICAL),
    SecretPattern("AWS_SECRET_ACCESS_KEY", re.compile(
        r"aws_secret_access_key\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})", re.IGNORECASE
    ), Severity.CRITICAL, group=1),
    SecretPattern("GITHUB_TOKEN", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_"
                                             r"[A-Za-z0-9_]{60,})\b"), Severity.CRITICAL),
    SecretPattern("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), Severity.HIGH),
    SecretPattern("STRIPE_SECRET_KEY", re.compile(r"\b[sr]k_live_[0-9A-Za-z]{24,}\b"),
                  Severity.CRITICAL),
    SecretPattern("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), Severity.HIGH),
    SecretPattern("LLM_API_KEY", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{32,}\b"),
                  Severity.CRITICAL),
    SecretPattern("JWT", re.compile(
        r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
    ), Severity.HIGH),
    SecretPattern("GENERIC_SECRET_ASSIGNMENT", re.compile(
        r"(?i)\b(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token|password|"
        r"passwd|pwd)\b\s*[:=]\s*['\"]?([^\s'\",;]{8,})"
    ), Severity.HIGH, group=1, min_entropy=3.0, literal_only=True),
]  # fmt: skip

TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_-]{24,}")
ENTROPY_THRESHOLD = 4.5  # bits per char; hex digests top out at 4.0


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    n = len(value)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def mask_secret(value: str) -> str:
    return value[:4] + "*" * min(max(len(value) - 4, 4), 12)


def _is_binary(path: Path) -> bool:
    with path.open("rb") as fh:
        return b"\x00" in fh.read(BINARY_SNIFF_BYTES)


class SecretsScanner(Scanner):
    """Scan text files (data or scripts) for credentials."""

    name = "secrets"

    def __init__(self, entropy: bool = True, max_examples: int = 1) -> None:
        self.entropy = entropy
        self.max_examples = max_examples

    def scan(self, target: ScanTarget) -> list[Finding]:
        path = target.path
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES or _is_binary(path):
            return []
        hits: dict[str, _Hit] = {}
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, start=1):
                for kind, severity, value in self._scan_line(line):
                    hit = hits.setdefault(kind, _Hit(severity=severity, first_line=line_no))
                    hit.count += 1
                    if len(hit.examples) < self.max_examples:
                        hit.examples.append(mask_secret(value))
        return [hit.finding(self.name, kind, target.label) for kind, hit in hits.items()]

    def _scan_line(self, line: str) -> Iterator[tuple[str, Severity, str]]:
        """Known patterns first; entropy only for tokens no pattern already claimed."""
        claimed: list[tuple[int, int]] = []
        for pattern in PATTERNS:
            for match in pattern.regex.finditer(line):
                value = match.group(pattern.group)
                if shannon_entropy(value) < pattern.min_entropy:
                    continue
                if pattern.literal_only and CODE_REFERENCE_RE.search(value):
                    continue
                claimed.append(match.span())
                yield pattern.name, pattern.severity, value
        if not self.entropy:
            return
        for match in TOKEN_RE.finditer(line):
            start, end = match.span()
            if any(s <= start < e or s < end <= e for s, e in claimed):
                continue
            token = match.group()
            if shannon_entropy(token) >= ENTROPY_THRESHOLD and _mixed(token):
                yield "HIGH_ENTROPY_STRING", Severity.MEDIUM, token


@dataclass
class _Hit:
    severity: Severity
    first_line: int
    count: int = 0
    examples: list[str] = field(default_factory=list)

    def finding(self, scanner: str, kind: str, label: str) -> Finding:
        return Finding(
            scanner=scanner,
            category=kind,
            severity=self.severity,
            location=f"{label}:{self.first_line}",
            count=self.count,
            message=f"{self.count} possible {kind} in {label}",
            masked_examples=self.examples,
            details={"first_line": self.first_line},
        )


def _mixed(token: str) -> bool:
    """Require letters and digits; long all-letter runs are usually identifiers or prose."""
    return any(c.isdigit() for c in token) and any(c.isalpha() for c in token)


def confirmed_secrets(findings: list[Finding]) -> list[Finding]:
    """Findings counted by the ``secrets_found_max`` policy rule (high severity and up)."""
    return [
        f
        for f in findings
        if f.scanner == SecretsScanner.name and at_least(f.severity, Severity.HIGH)
    ]
