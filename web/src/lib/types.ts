import type { Role, State } from "./platform";

export type Me = {
  id: string;
  email: string;
  name: string;
  super_admin: boolean;
  memberships: { tenant: string; role: Role }[];
};

export type Tenant = {
  id: string;
  slug: string;
  name: string;
  plan: "free" | "team" | "enterprise";
  status: "active" | "suspended";
  created_at: string;
};

export type Member = { user_id: string; email: string; name: string; role: Role };

export type Dataset = {
  id: string;
  name: string;
  source: string;
  license: string;
  consent: string;
  sha256: string;
  size_bytes: number;
  records: number;
  entities: number;
  created_at: string;
};

export type Version = {
  id: string;
  model_id: string;
  version: string;
  state: State;
  parent_id: string | null;
  dataset_id: string | null;
  reference_dataset_id: string | null;
  artifact_sha256: string | null;
  verdict: string | null;
  auc: number | null;
  critical: number;
  high: number;
  attestation_sha256: string | null;
  created_at: string;
  updated_at: string;
};

export type Model = {
  id: string;
  name: string;
  access: "open-weight" | "api";
  base: string;
  description: string;
  created_at: string;
};

export type ModelDetail = Model & { versions: Version[] };

export type Finding = {
  record: string;
  entity_type: string;
  masked_value: string;
  risk: number;
  severity: "info" | "low" | "medium" | "high" | "critical";
  q_value: number;
  confirmed: boolean | null;
  exposure: number | null;
  extraction_rate: number | null;
};

export type Audit = {
  model: string;
  access: string;
  primary_method: string;
  entities_audited: number;
  auc: number | null;
  tpr_at_fpr: Record<string, number>;
  severity_counts: Record<string, number>;
  metrics: { method: string; auc: number; entity_type: string | null }[];
  findings: Finding[];
  taxonomy: string[];
};

export type Job = {
  id: string;
  kind: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  payload: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string | null;
  attempts: number;
  created_at: string;
  finished_at: string | null;
};

export type AuditEvent = {
  seq: number;
  actor: string;
  action: string;
  target_type: string;
  target_id: string;
  details: Record<string, unknown>;
  hash: string;
  created_at: string;
};

export type Deployment = {
  id: string;
  version_id: string;
  model: string;
  version: string;
  environment: "dev" | "consumer";
  status: string;
  endpoint: string;
  created_at: string;
};

export type TestReport = {
  id: string;
  prompt_count: number;
  leaks_found: number;
  summary: string;
  created_at: string;
};
