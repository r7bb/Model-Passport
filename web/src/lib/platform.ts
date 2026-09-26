/**
 * Pure helpers shared by pages and tests: roles, permissions, lifecycle steps, and formatting.
 * They mirror the backend (model_passport.platform.rbac and .lifecycle); the backend still
 * re-checks every action, so these only decide what the UI offers.
 */

export type Role =
  | "org_admin"
  | "ml_engineer"
  | "compliance_auditor"
  | "canary_tester"
  | "end_consumer"
  | "external_reviewer";

export type State =
  | "registered"
  | "auditing"
  | "findings"
  | "clean"
  | "remediating"
  | "superseded"
  | "canary"
  | "verifying"
  | "approved"
  | "released"
  | "rolled_back"
  | "rejected"
  | "killed";

export const ROLE_LABELS: Record<Role, string> = {
  org_admin: "Org admin",
  ml_engineer: "ML engineer",
  compliance_auditor: "Compliance auditor",
  canary_tester: "Canary tester",
  end_consumer: "End consumer",
  external_reviewer: "External reviewer",
};

type Permission =
  | "models.read"
  | "models.write"
  | "audits.run"
  | "remediation.run"
  | "findings.read"
  | "approvals.decide"
  | "releases.manage"
  | "killswitch.activate"
  | "endpoints.dev"
  | "testreports.submit"
  | "members.manage"
  | "auditlog.read"
  | "analytics.read"
  | "provenance.read"
  | "reports.read"
  | "jobs.read";

const READ: Permission[] = [
  "models.read",
  "findings.read",
  "analytics.read",
  "provenance.read",
  "jobs.read",
];

const PERMISSIONS: Record<Role, Permission[]> = {
  org_admin: [
    ...READ,
    "members.manage",
    "approvals.decide",
    "releases.manage",
    "killswitch.activate",
    "auditlog.read",
    "reports.read",
  ],
  ml_engineer: [...READ, "models.write", "audits.run", "remediation.run", "endpoints.dev"],
  compliance_auditor: [
    ...READ,
    "approvals.decide",
    "killswitch.activate",
    "auditlog.read",
    "reports.read",
  ],
  canary_tester: ["models.read", "endpoints.dev", "testreports.submit"],
  end_consumer: [],
  external_reviewer: ["reports.read", "provenance.read"],
};

export function can(role: Role | null, permission: Permission, superAdmin = false): boolean {
  return superAdmin || (role !== null && PERMISSIONS[role].includes(permission));
}

/** The dashboard modules from the product flow, and who sees each. */
export const MODULES: { id: string; label: string; path: string; needs: Permission }[] = [
  { id: "M1", label: "Dashboard", path: "", needs: "models.read" },
  { id: "M2", label: "Access & roles", path: "/members", needs: "members.manage" },
  { id: "M3", label: "Analytics", path: "/analytics", needs: "analytics.read" },
  { id: "M4", label: "Audit log", path: "/audit-log", needs: "auditlog.read" },
  { id: "M5", label: "Models", path: "/models", needs: "models.read" },
  { id: "M7", label: "Pipelines & workers", path: "/jobs", needs: "jobs.read" },
  { id: "M8", label: "Dev endpoints & testing", path: "/deployments", needs: "endpoints.dev" },
  { id: "M9", label: "Data provenance", path: "/data", needs: "provenance.read" },
];

export function visibleModules(role: Role | null, superAdmin = false) {
  return MODULES.filter((m) => can(role, m.needs, superAdmin));
}

export type Step = {
  label: string;
  kind: "audit" | "remediate" | "transition" | "approve" | "reject" | "kill";
  to?: State;
  danger?: boolean;
};

/** The next lifecycle steps this role can take from a version's state (product flow). */
export function nextSteps(state: State, role: Role | null, superAdmin = false): Step[] {
  const steps: Step[] = [];
  const allow = (p: Permission) => can(role, p, superAdmin);
  if (["registered", "clean", "findings"].includes(state) && allow("audits.run")) {
    steps.push({ label: state === "registered" ? "Run audit" : "Re-audit", kind: "audit" });
  }
  if (state === "findings" && allow("remediation.run")) {
    steps.push({ label: "Remediate", kind: "remediate" });
  }
  if (state === "clean" && allow("endpoints.dev")) {
    steps.push({ label: "Deploy to canary", kind: "transition", to: "canary" });
  }
  if (state === "canary" && allow("endpoints.dev")) {
    steps.push({ label: "Start verification", kind: "transition", to: "verifying" });
  }
  if (state === "verifying" && allow("approvals.decide")) {
    steps.push({ label: "Approve", kind: "approve" });
    steps.push({ label: "Reject", kind: "reject", danger: true });
    steps.push({ label: "Send back to remediation", kind: "transition", to: "findings" });
  }
  if (state === "approved" && allow("approvals.decide")) {
    steps.push({ label: "Withdraw approval", kind: "transition", to: "rejected", danger: true });
  }
  if (state === "approved" && allow("releases.manage")) {
    steps.push({ label: "Release to consumers", kind: "transition", to: "released" });
  }
  if (state === "released" && allow("releases.manage")) {
    steps.push({ label: "Mark rolled back", kind: "transition", to: "rolled_back" });
  }
  const killable = ["findings", "clean", "canary", "verifying", "approved", "released"];
  if (killable.includes(state) && allow("killswitch.activate")) {
    steps.push({ label: "Kill switch", kind: "kill", danger: true });
  }
  return steps;
}

export const STATE_TONE: Record<State, "neutral" | "good" | "warn" | "bad" | "info"> = {
  registered: "neutral",
  auditing: "info",
  findings: "bad",
  clean: "good",
  remediating: "info",
  superseded: "neutral",
  canary: "info",
  verifying: "info",
  approved: "good",
  released: "good",
  rolled_back: "warn",
  rejected: "bad",
  killed: "bad",
};

/** "usps" for "usps.mp.com" when the base domain is "mp.com"; null otherwise. */
export function tenantFromHost(host: string | null, baseDomain: string): string | null {
  if (!host || !baseDomain) return null;
  const hostname = host.split(":")[0].toLowerCase();
  const suffix = `.${baseDomain.toLowerCase()}`;
  if (!hostname.endsWith(suffix)) return null;
  const slug = hostname.slice(0, -suffix.length);
  return /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/.test(slug) ? slug : null;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
}

export function formatAuc(auc: number | null | undefined): string {
  return auc === null || auc === undefined ? "—" : auc.toFixed(3);
}

/** The version to suggest next: one minor step past the newest ("1.2.0" -> "1.3.0"). */
export function nextVersion(versions: string[]): string {
  let best: number[] | null = null;
  for (const v of versions) {
    const parts = /^(\d+)\.(\d+)\.(\d+)$/.exec(v)?.slice(1).map(Number);
    if (parts && (!best || parts[0] > best[0] || (parts[0] === best[0] && parts[1] > best[1]))) best = parts;
  }
  return best ? `${best[0]}.${best[1] + 1}.0` : "1.0.0";
}
