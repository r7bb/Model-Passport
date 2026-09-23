"""Render captured CLI output as a terminal-window PNG (used for README images).

    passport build 2>&1 | python scripts/terminal_shot.py - docs/images/cli-build.png \\
        --title "passport build"
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

GREEN, YELLOW, RED, BLUE = "#4ade80", "#facc15", "#f87171", "#93c5fd"
GOOD = r"\[(PASS|ok\s*)\]|\bVERIFIED\b|\bverdict: PASS\b|no retraining needed"
BAD = (
    r"\[FAIL\]|\[DRIFT\]|\bverdict: FAIL\b|VERIFICATION FAILED|CHANGED|INVALID|"
    r"retraining recommended|MISMATCH"
)
HIGHLIGHTS = [
    (re.compile(GOOD), GREEN),
    (re.compile(r"\[WARN\]"), YELLOW),
    (re.compile(BAD), RED),
    (re.compile(r"^\$ .*$", re.MULTILINE), BLUE),
]

PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin: 0; padding: 24px; background: #e5e7eb; font-family: -apple-system, sans-serif; }}
.win {{ background: #0f172a; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,.25);
        overflow: hidden; display: inline-block; min-width: 900px; }}
.bar {{ background: #1e293b; padding: 10px 14px; color: #94a3b8; font-size: 13px; }}
.dot {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%;
        margin-right: 6px; vertical-align: middle; }}
pre {{ margin: 0; padding: 16px 20px; color: #e2e8f0; font: 13px/1.5 ui-monospace, Menlo,
       monospace; white-space: pre; }}
</style></head><body><div class="win"><div class="bar">
<span class="dot" style="background:#f87171"></span><span class="dot" style="background:#facc15">
</span><span class="dot" style="background:#4ade80"></span>&nbsp; {title}</div>
<pre>{body}</pre></div></body></html>"""


def colorize(text: str) -> str:
    escaped = html.escape(text)
    for pattern, color in HIGHLIGHTS:
        escaped = pattern.sub(
            lambda m, c=color: f'<span style="color:{c};font-weight:600">{m.group(0)}</span>',
            escaped,
        )
    return escaped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="text file with captured output, or - for stdin")
    parser.add_argument("out", type=Path)
    parser.add_argument("--title", default="terminal")
    args = parser.parse_args()

    text = sys.stdin.read() if args.source == "-" else Path(args.source).read_text()
    page_html = PAGE.format(title=html.escape(args.title), body=colorize(text.rstrip()))
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
        fh.write(page_html)
        page_path = Path(fh.name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1000, "height": 400}, device_scale_factor=2)
        page.goto(page_path.as_uri())
        page.locator(".win").screenshot(path=str(args.out))
        browser.close()
    page_path.unlink()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
