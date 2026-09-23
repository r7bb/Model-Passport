"""Capture a full-page PNG of a URL or local HTML file (used for README images).

python scripts/screenshot.py passport.html docs/images/report.png --width 1200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="URL or path to an HTML file")
    parser.add_argument("out", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--full-page", action="store_true")
    parser.add_argument("--wait-for", help="CSS selector or text to wait for", default=None)
    parser.add_argument("--delay-ms", type=int, default=500)
    parser.add_argument("--clip-height", type=int, default=None)
    parser.add_argument("--click", help="text of a tab or button to click before capturing")
    args = parser.parse_args()

    target = args.target
    if "://" not in target:
        target = Path(target).resolve().as_uri()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(
            viewport={"width": args.width, "height": args.height}, device_scale_factor=2
        )
        page.goto(target, wait_until="networkidle")
        if args.wait_for:
            page.get_by_text(args.wait_for).first.wait_for(timeout=60_000)
        if args.click:
            page.get_by_role("tab", name=args.click).or_(page.get_by_text(args.click)).first.click()
        page.wait_for_timeout(args.delay_ms)
        clip = None
        if args.clip_height:
            clip = {"x": 0, "y": 0, "width": args.width, "height": args.clip_height}
        page.screenshot(path=str(args.out), full_page=args.full_page and clip is None, clip=clip)
        browser.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
