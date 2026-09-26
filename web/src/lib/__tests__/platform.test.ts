import { describe, expect, it } from "vitest";

import { can, nextSteps, nextVersion, tenantFromHost, visibleModules } from "../platform";

describe("tenantFromHost", () => {
  it("reads the organization from its subdomain", () => {
    expect(tenantFromHost("usps.mp.example", "mp.example")).toBe("usps");
    expect(tenantFromHost("USPS.mp.example:3000", "mp.example")).toBe("usps");
  });
  it("ignores other hosts and invalid slugs", () => {
    expect(tenantFromHost("mp.example", "mp.example")).toBeNull();
    expect(tenantFromHost("a.b.mp.example", "mp.example")).toBeNull();
    expect(tenantFromHost("evil.example", "mp.example")).toBeNull();
    expect(tenantFromHost("usps.mp.example", "")).toBeNull();
    expect(tenantFromHost(null, "mp.example")).toBeNull();
  });
});

describe("roles", () => {
  it("mirrors the backend permissions", () => {
    expect(can("ml_engineer", "audits.run")).toBe(true);
    expect(can("ml_engineer", "approvals.decide")).toBe(false);
    expect(can("compliance_auditor", "killswitch.activate")).toBe(true);
    expect(can("end_consumer", "models.read")).toBe(false);
    expect(can(null, "members.manage", true)).toBe(true);
  });
  it("shows each role only its modules", () => {
    const ids = (role: Parameters<typeof visibleModules>[0]) => visibleModules(role).map((m) => m.id);
    expect(ids("org_admin")).toContain("M2");
    expect(ids("ml_engineer")).not.toContain("M2");
    expect(ids("canary_tester")).toEqual(["M1", "M5", "M8"]);
    expect(ids("external_reviewer")).toEqual(["M9"]);
    expect(ids("end_consumer")).toEqual([]);
  });
});

describe("nextSteps", () => {
  const labels = (...args: Parameters<typeof nextSteps>) => nextSteps(...args).map((s) => s.label);
  it("follows detect, remediate, verify and deploy", () => {
    expect(labels("registered", "ml_engineer")).toEqual(["Run audit"]);
    expect(labels("findings", "ml_engineer")).toEqual(["Re-audit", "Remediate"]);
    expect(labels("clean", "ml_engineer")).toContain("Deploy to canary");
    expect(labels("verifying", "compliance_auditor")).toContain("Approve");
    expect(labels("approved", "org_admin")).toContain("Release to consumers");
  });
  it("offers the kill switch only to roles that hold it", () => {
    expect(labels("released", "compliance_auditor")).toContain("Kill switch");
    expect(labels("released", "ml_engineer")).not.toContain("Kill switch");
    expect(labels("killed", "org_admin")).toEqual([]);
  });
  it("never offers system-only states", () => {
    expect(labels("auditing", "org_admin")).toEqual([]);
    expect(labels("remediating", "ml_engineer")).toEqual([]);
  });
});

describe("nextVersion", () => {
  it("suggests the next minor version", () => {
    expect(nextVersion([])).toBe("1.0.0");
    expect(nextVersion(["1.0.0", "1.2.0", "1.1.0"])).toBe("1.3.0");
    expect(nextVersion(["1.9.0", "2.0.1"])).toBe("2.1.0");
    expect(nextVersion(["custom"])).toBe("1.0.0");
  });
});
