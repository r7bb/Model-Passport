"""Artifact safety: pickle opcode scanning and dependency vulnerability audit."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import joblib
from picklescan.scanner import SafetyLevel, scan_file_path

from model_passport.core.schema import DependencyAudit, Finding, Severity

PICKLE_SUFFIXES = {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".bin", ".npy"}
SAFE_SUFFIXES = {".safetensors", ".onnx", ".json", ".txt", ".ubj"}
OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{id}"
GHSA_SEVERITY = {
    "LOW": Severity.LOW,
    "MODERATE": Severity.MEDIUM,
    "MEDIUM": Severity.MEDIUM,
    "HIGH": Severity.HIGH,
    "CRITICAL": Severity.CRITICAL,
}


class UnsafeArtifactError(Exception):
    """Raised when refusing to deserialize an artifact."""


# --- Pickle scanning -----------------------------------------------------------------------


def is_pickle_like(path: Path) -> bool:
    return path.suffix.lower() in PICKLE_SUFFIXES


def scan_model_file(path: Path, label: str | None = None) -> list[Finding]:
    """Scan a serialized model. Dangerous pickle imports are critical; raw pickle is low."""
    label = label or path.name
    suffix = path.suffix.lower()
    if suffix in SAFE_SUFFIXES:
        return [
            Finding(
                scanner="artifact", category="SAFE_FORMAT", severity=Severity.INFO,
                location=label, message=f"{suffix} does not execute code on load",
            )
        ]  # fmt: skip
    if not is_pickle_like(path):
        return [
            Finding(
                scanner="artifact", category="UNKNOWN_FORMAT", severity=Severity.LOW,
                location=label, message=f"cannot verify load safety of {suffix or 'no suffix'}",
            )
        ]  # fmt: skip

    result = scan_file_path(str(path))
    if result.scan_err:
        return [
            Finding(
                scanner="artifact", category="SCAN_ERROR", severity=Severity.HIGH,
                location=label, message="pickle could not be parsed; treat as unsafe",
            )
        ]  # fmt: skip

    dangerous = sorted(
        {f"{g.module}.{g.name}" for g in result.globals if g.safety == SafetyLevel.Dangerous}
    )
    unknown_modules = sorted(
        {g.module.split(".")[0] for g in result.globals if g.safety == SafetyLevel.Suspicious}
    )
    findings = []
    if dangerous:
        findings.append(
            Finding(
                scanner="artifact",
                category="UNSAFE_PICKLE",
                severity=Severity.CRITICAL,
                location=label,
                count=len(dangerous),
                message=f"pickle imports dangerous callables: {', '.join(dangerous)}",
                details={"dangerous_globals": dangerous},
            )
        )
    findings.append(
        Finding(
            scanner="artifact",
            category="RAW_PICKLE_FORMAT",
            severity=Severity.LOW,
            location=label,
            count=len(result.globals),
            message="raw pickle executes code on load; prefer safetensors, ONNX, or skops",
            details={"imported_modules": unknown_modules},
        )
    )
    return findings


def safe_load_pickle(path: Path) -> Any:
    """Unpickle only after the scan finds no dangerous imports."""
    findings = scan_model_file(path)
    blocking = [f for f in findings if f.category in {"UNSAFE_PICKLE", "SCAN_ERROR"}]
    if blocking:
        raise UnsafeArtifactError(f"refusing to load {path.name}: {blocking[0].message}")
    if path.suffix.lower() == ".joblib":
        return joblib.load(path)
    with path.open("rb") as fh:
        return pickle.load(fh)  # noqa: S301 - opcode scan above found no dangerous imports


# --- Dependency audit ----------------------------------------------------------------------


def _ghsa_id(vuln: dict[str, Any]) -> str | None:
    for candidate in [vuln.get("id", ""), *vuln.get("aliases", [])]:
        if str(candidate).startswith("GHSA-"):
            return str(candidate)
    return None


def fetch_severity(vuln_id: str, timeout: float = 10.0) -> Severity | None:
    """Severity from the GitHub advisory record in OSV, or None if unavailable."""
    try:
        url = OSV_VULN_URL.format(id=vuln_id)  # constant https:// URL
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
    label = str(data.get("database_specific", {}).get("severity", "")).upper()
    return GHSA_SEVERITY.get(label)


def run_pip_audit(
    dependencies: dict[str, str], timeout: float = 300.0
) -> tuple[DependencyAudit, str, list[dict[str, Any]]]:
    """Audit pinned ``name==version`` dependencies. Returns (status, message, raw results)."""
    pins = [f"{name}=={version}" for name, version in sorted(dependencies.items())]
    if not pins:
        return DependencyAudit.OK, "no dependencies", []
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "requirements.txt"
        req.write_text("\n".join(pins) + "\n", encoding="utf-8")
        cmd = [
            sys.executable, "-m", "pip_audit", "-r", str(req), "--no-deps", "--disable-pip",
            "--progress-spinner", "off", "-f", "json",
        ]  # fmt: skip
        try:
            # pip-audit exits 1 when it finds vulnerabilities, so the code is not an error.
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return DependencyAudit.ERROR, f"pip-audit did not run: {exc}", []
    if "No module named pip_audit" in result.stderr:
        return DependencyAudit.ERROR, "pip-audit is not installed (pip install pip-audit)", []
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        tail = (result.stderr.strip().splitlines() or ["no output"])[-1]
        return DependencyAudit.ERROR, f"pip-audit failed: {tail}", []
    return DependencyAudit.OK, f"audited {len(pins)} packages", data.get("dependencies", [])


def audit_dependencies(
    dependencies: dict[str, str], lookup_severity: bool = True
) -> tuple[DependencyAudit, str, list[Finding]]:
    ignore = {"model-passport", "model_passport"}
    deps = {k: v for k, v in dependencies.items() if k.lower() not in ignore}
    status, message, results = run_pip_audit(deps)
    vulns: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for dep in results:
        for vuln in dep.get("vulns", []):
            vulns.setdefault(vuln["id"], (dep["name"], dep["version"], vuln))

    ghsa = {vid: _ghsa_id(v) for vid, (_, _, v) in vulns.items()}
    severities: dict[str, Severity | None] = {}
    if lookup_severity:
        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = [g for g in set(ghsa.values()) if g]
            severities = dict(zip(ids, pool.map(fetch_severity, ids), strict=True))

    findings = []
    for vid, (name, version, vuln) in sorted(vulns.items()):
        severity = severities.get(ghsa[vid] or "")
        cves = [a for a in [vid, *vuln.get("aliases", [])] if a.startswith("CVE-")]
        fixed_in = ", ".join(vuln.get("fix_versions", [])) or "no release"
        findings.append(
            Finding(
                scanner="pip_audit",
                category=cves[0] if cves else vid,
                severity=severity or Severity.MEDIUM,
                location=f"{name}=={version}",
                count=1,
                message=f"{name} {version}: {vid}; fixed in {fixed_in}",
                details={
                    "id": vid,
                    "aliases": vuln.get("aliases", []),
                    "fix_versions": vuln.get("fix_versions", []),
                    "severity_source": "ghsa" if severity else "unknown",
                },
            )
        )
    return status, message, findings
