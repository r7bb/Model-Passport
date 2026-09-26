"""Capture README screenshots of the web app, signed in as each role.

Run ``scripts/demo_platform.py`` and the web app first, then::

    python scripts/web_screenshots.py --web http://localhost:3000 --api http://localhost:8080
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
from playwright.sync_api import Browser, Page, sync_playwright

PASSWORD = "demo password 2026"  # noqa: S105 - the local demo accounts
ORG = "northwind"
OUT = Path(__file__).resolve().parents[1] / "docs" / "images"


def _ids(api: str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Emails by role, model ids by name, and version ids by "model@version"."""
    http = httpx.Client(base_url=f"{api}/api/v1")
    login = {"email": f"admin@{ORG}.example", "password": PASSWORD}
    token = http.post("/auth/login", json=login).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}", "X-MP-Tenant": ORG}
    emails = {m["role"]: m["email"] for m in http.get("/members", headers=headers).json()}
    models = {m["name"]: m["id"] for m in http.get("/models", headers=headers).json()}
    versions = {}
    for name, model_id in models.items():
        for v in http.get(f"/models/{model_id}", headers=headers).json()["versions"]:
            versions[f"{name}@{v['version']}"] = v["id"]
    return emails, models, versions


def _session(browser: Browser, web: str, email: str) -> Page:
    page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
    page.goto(f"{web}/login")
    page.fill("input[name=email]", email)
    page.fill("input[name=password]", PASSWORD)
    page.click("button[type=submit]")
    page.wait_for_url(lambda url: "/login" not in url)
    return page


def _shot(page: Page, web: str, path: str, name: str, full: bool = False) -> None:
    page.goto(f"{web}{path}", wait_until="networkidle")
    page.wait_for_timeout(300)
    page.screenshot(path=str(OUT / name), full_page=full)
    print(f"wrote docs/images/{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", default="http://localhost:3000")
    parser.add_argument("--api", default="http://localhost:8080")
    args = parser.parse_args()
    emails, models, versions = _ids(args.api)
    web, org = args.web, f"/o/{ORG}"
    assistant = models["support-assistant"]
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        login = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        _shot(login, web, "/login", "web-login.png")

        root = _session(browser, web, "root@platform.example")
        _shot(root, web, "/admin", "web-admin.png")

        admin = _session(browser, web, emails["org_admin"])
        _shot(admin, web, org, "web-dashboard.png")
        _shot(admin, web, f"{org}/members", "web-members.png")

        auditor = _session(browser, web, emails["compliance_auditor"])
        _shot(auditor, web, f"{org}/analytics", "web-analytics.png")
        _shot(auditor, web, f"{org}/audit-log", "web-audit-log.png")

        engineer = _session(browser, web, emails["ml_engineer"])
        _shot(engineer, web, f"{org}/models/{assistant}", "web-model.png")
        risky = versions["claims-summarizer@1.0.0"]
        _shot(engineer, web, f"{org}/versions/{risky}", "web-findings.png")
        _shot(engineer, web, f"{org}/deployments", "web-deployments.png")
        _shot(engineer, web, f"{org}/data", "web-data.png")

        reviewer = _session(browser, web, emails["external_reviewer"])
        _shot(reviewer, web, f"{org}/models/{assistant}/report", "web-diligence.png")
        browser.close()


if __name__ == "__main__":
    main()
