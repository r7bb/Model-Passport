import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

const APP = join(__dirname, "..", "..", "app");

function pages(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return pages(path);
    return name === "page.tsx" ? [path] : [];
  });
}

describe("page metadata", () => {
  it("gives every page its own title", () => {
    const untitled = pages(APP).filter(
      (file) => !/export const metadata: Metadata = \{ title: "[^"]+" \}/.test(readFileSync(file, "utf8")),
    );
    expect(untitled).toEqual([]);
  });

  it("keeps the private console out of search engines and describes it in 120 to 160 characters", () => {
    const layout = readFileSync(join(APP, "layout.tsx"), "utf8");
    expect(layout).toContain("robots: { index: false, follow: false }");
    expect(layout).toContain('template: "%s · Model Passport"');
    const description = /description:\s*"([^"]+)"/.exec(layout)?.[1] ?? "";
    expect(description.length).toBeGreaterThanOrEqual(120);
    expect(description.length).toBeLessThanOrEqual(160);
  });

  it("has a custom not-found page", () => {
    expect(existsSync(join(APP, "not-found.tsx"))).toBe(true);
  });
});
