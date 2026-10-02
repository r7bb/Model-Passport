import { describe, expect, it } from "vitest";

import nextConfig from "../../../next.config";

async function headerMap(): Promise<Map<string, string>> {
  const rules = (await nextConfig.headers?.()) ?? [];
  const all = rules.find((rule) => rule.source === "/:path*");
  return new Map((all?.headers ?? []).map((h) => [h.key.toLowerCase(), h.value]));
}

describe("security headers", () => {
  it("tells browsers to use HTTPS for two years, including subdomains", async () => {
    const hsts = (await headerMap()).get("strict-transport-security") ?? "";
    const maxAge = Number(/max-age=(\d+)/.exec(hsts)?.[1] ?? 0);
    expect(maxAge).toBeGreaterThanOrEqual(31_536_000);
    expect(hsts).toContain("includeSubDomains");
    expect(hsts).not.toContain("preload");
  });

  it("sends a Content-Security-Policy in report-only mode, not enforced yet", async () => {
    const headers = await headerMap();
    const csp = headers.get("content-security-policy-report-only") ?? "";
    expect(csp).toContain("default-src 'self'");
    expect(csp).toContain("frame-ancestors 'none'");
    expect(csp).toContain("object-src 'none'");
    expect(headers.has("content-security-policy")).toBe(false);
  });

  it("keeps the existing hardening headers", async () => {
    const headers = await headerMap();
    expect(headers.get("x-frame-options")).toBe("DENY");
    expect(headers.get("x-content-type-options")).toBe("nosniff");
    expect(headers.get("referrer-policy")).toBe("same-origin");
  });
});
