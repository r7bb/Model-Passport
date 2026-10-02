import type { GuardEvent, GuardReport, GuardStats } from "./types";

/** Helpers for the MP Guard screens: pure, so the pages stay thin and these stay tested. */

type Day = GuardStats["per_day"][number];

const DAY_MS = 24 * 60 * 60 * 1000;

/** The last `days` days (UTC, oldest first, ending today), with quiet days as zero requests. */
export function fillDays(perDay: Day[], days: number, today: Date = new Date()): Day[] {
  const byDay = new Map(perDay.map((d) => [d.day, d]));
  return Array.from({ length: days }, (_, i) => {
    const day = new Date(today.getTime() - (days - 1 - i) * DAY_MS).toISOString().slice(0, 10);
    return byDay.get(day) ?? { day, requests: 0 };
  });
}

/** The two lines an app changes to route its OpenAI SDK through MP Guard. */
export function quickstart(apiUrl: string, endpoint: string): string {
  const base = `${apiUrl.replace(/\/+$/, "")}${endpoint}`;
  return [
    "from openai import OpenAI",
    "",
    "client = OpenAI(",
    `    base_url="${base}",`,
    '    api_key="mpk_…",  # an MP Guard key from API keys',
    ")",
  ].join("\n");
}

export function total(counts: Record<string, number> | undefined): number {
  return Object.values(counts ?? {}).reduce((a, b) => a + b, 0);
}

const REPORT_LABELS: [keyof GuardReport, string][] = [
  ["masked", "kept"],
  ["redacted", "redacted"],
  ["leaked", "leaked"],
  ["memorized", "memorized"],
];

/** One line on what the guard did to a request, for the activity table. */
export function describeReport(report: GuardReport): string {
  const parts = REPORT_LABELS.flatMap(([kind, label]) => {
    const counts = report[kind];
    const n = total(counts);
    return n ? [`${label} ${n} (${Object.keys(counts ?? {}).join(", ")})`] : [];
  });
  return parts.join(" · ") || "nothing sensitive";
}

export const OUTCOME_TONE: Record<GuardEvent["outcome"], "good" | "info" | "bad" | "warn"> = {
  passed: "good",
  protected: "info",
  blocked: "bad",
  error: "warn",
};
