"""Render a passport as a self-contained HTML report."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, Undefined, select_autoescape

from model_passport.core.schema import Passport, Severity, at_least
from model_passport.core.verifier import VerificationReport
from model_passport.report.summary import summarize


def _env() -> Environment:
    env = Environment(
        loader=PackageLoader("model_passport.report", "templates"),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["short"] = lambda value, n=12: (str(value)[:n] + "…") if value else ""
    env.filters["pct"] = lambda value: "" if _missing(value) else f"{value:.1%}"
    env.filters["num"] = lambda value: "" if _missing(value) else f"{value:,.4g}"
    env.filters["threshold"] = format_threshold
    env.tests["serious"] = lambda finding: at_least(finding.severity, Severity.HIGH)
    return env


def _missing(value: object) -> bool:
    return value is None or isinstance(value, Undefined)


def format_threshold(value: object) -> str:
    """``{"warn": 0.55, "fail": 0.6}`` -> ``warn 0.55 · fail 0.6``."""
    if isinstance(value, dict):
        return " · ".join(f"{level} {limit}" for level, limit in value.items())
    return str(value)


def render_html(passport: Passport, verification: VerificationReport | None = None) -> str:
    template = _env().get_template("passport.html.j2")
    return template.render(p=passport, summary=summarize(passport), verification=verification)


def write_html(
    passport: Passport, out: Path, verification: VerificationReport | None = None
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(passport, verification), encoding="utf-8")
