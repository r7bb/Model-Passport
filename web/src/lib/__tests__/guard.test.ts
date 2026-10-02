import { describe, expect, it } from "vitest";

import { describeReport, fillDays, quickstart, total } from "../guard";

describe("fillDays", () => {
  it("returns one entry per day, oldest first, ending today", () => {
    const days = fillDays([{ day: "2026-10-01", requests: 3, blocked: 1 }], 3, new Date("2026-10-02T12:00:00Z"));
    expect(days.map((d) => d.day)).toEqual(["2026-09-30", "2026-10-01", "2026-10-02"]);
    expect(days[1]).toEqual({ day: "2026-10-01", requests: 3, blocked: 1 });
  });
  it("fills quiet days with zero requests", () => {
    const days = fillDays([], 2, new Date("2026-10-02T00:30:00Z"));
    expect(days).toEqual([
      { day: "2026-10-01", requests: 0 },
      { day: "2026-10-02", requests: 0 },
    ]);
  });
  it("ignores days outside the window", () => {
    const days = fillDays([{ day: "2026-01-01", requests: 9 }], 1, new Date("2026-10-02T12:00:00Z"));
    expect(days).toEqual([{ day: "2026-10-02", requests: 0 }]);
  });
});

describe("quickstart", () => {
  it("points the OpenAI SDK at the guard endpoint", () => {
    const code = quickstart("https://api.example.com/", "/guard/v1");
    expect(code).toContain('base_url="https://api.example.com/guard/v1"');
    expect(code).toContain('api_key="mpk_');
    expect(code).toContain("from openai import OpenAI");
  });
});

describe("total", () => {
  it("adds every count, treating a missing map as zero", () => {
    expect(total({ EMAIL: 2, PHONE: 1 })).toBe(3);
    expect(total(undefined)).toBe(0);
  });
});

describe("describeReport", () => {
  it("lists what was protected, by type", () => {
    expect(describeReport({ masked: { EMAIL: 2, PHONE: 1 }, memorized: { SSN: 1 } })).toBe(
      "kept 3 (EMAIL, PHONE) · memorized 1 (SSN)",
    );
  });
  it("says so when nothing was found", () => {
    expect(describeReport({})).toBe("nothing sensitive");
    expect(describeReport({ masked: {} })).toBe("nothing sensitive");
  });
});
