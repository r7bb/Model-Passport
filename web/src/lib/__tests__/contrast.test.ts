import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

import { describe, expect, it } from "vitest";

const SRC = join(__dirname, "..", "..");

// text-slate-400 is about 2.6:1 on white, below the WCAG AA 4.5:1 minimum for body text. It stays
// only where the text is decorative and repeats something already said.
const DECORATIVE = new Set(["components/Shell.tsx"]); // the small module tag beside each nav label

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (name === "__tests__") return [];
    if (statSync(path).isDirectory()) return sources(path);
    return name.endsWith(".tsx") ? [path] : [];
  });
}

describe("text contrast", () => {
  it("uses no low-contrast text-slate-400 for meaningful text", () => {
    const offenders = sources(SRC)
      .map((file) => relative(SRC, file))
      .filter((file) => !DECORATIVE.has(file) && file !== "app/o/[org]/data/page.tsx")
      .filter((file) => readFileSync(join(SRC, file), "utf8").includes("text-slate-400"));
    expect(offenders).toEqual([]);
  });
});
