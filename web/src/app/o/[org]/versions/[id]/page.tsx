import { ActionForm } from "@/components/ActionForm";
import { AutoRefresh } from "@/components/AutoRefresh";
import { Lifecycle } from "@/components/Lifecycle";
import { Badge, Card, Field, Notice, PageHeader, SeverityBadge, Stat, StateBadge, Table, Td, TextLink, inputClass } from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import { can, formatAuc, formatDate } from "@/lib/platform";
import { load, requireMember } from "@/lib/session";
import type { Audit, TestReport, Version } from "@/lib/types";

import { submitTestReport } from "../../actions";

async function findingsOf(org: string, id: string): Promise<Audit | null> {
  try {
    return (await api<{ audit: Audit }>(`/versions/${id}/findings`, { org })).audit;
  } catch (error) {
    if (error instanceof ApiError && (error.status === 404 || error.status === 403)) return null;
    throw error;
  }
}

/** M5: one version's entity audit (masked values only), canary reports, and next steps. */
const PAGE = 50;
const SEVERITIES = ["critical", "high", "medium", "low"] as const;

export default async function VersionPage({ params, searchParams }: PageProps<"/o/[org]/versions/[id]">) {
  const { org, id } = await params;
  const showAll = (await searchParams).show === "all";
  const { me, role } = await requireMember(org);
  const safe = encodeURIComponent(id);
  const version = await load(api<Version>(`/versions/${safe}`, { org }));
  const [audit, reports] = await Promise.all([
    findingsOf(org, safe),
    load(api<TestReport[]>(`/versions/${safe}/test-reports`, { org })),
  ]);
  const testing = version.state === "canary" || version.state === "verifying";
  const leaks = reports.reduce((n, r) => n + r.leaks_found, 0);
  return (
    <>
      <AutoRefresh active={version.state === "auditing" || version.state === "remediating"} />
      <PageHeader
        title={`Version ${version.version}`}
        subtitle={
          <>
            <StateBadge state={version.state} />{" "}
            <TextLink href={`/o/${org}/models/${version.model_id}`}>back to model</TextLink>
          </>
        }
        actions={<Lifecycle org={org} versionId={version.id} state={version.state} role={role} superAdmin={me.super_admin} />}
      />
      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Attack AUC" value={formatAuc(version.auc)} />
        <Stat label="Critical" value={version.critical} tone={version.critical ? "bad" : "good"} />
        <Stat label="High" value={version.high} tone={version.high ? "bad" : "good"} />
        <Stat label="Canary leaks" value={leaks} tone={leaks ? "bad" : reports.length ? "good" : undefined} />
      </div>
      {version.attestation_sha256 ? (
        <div className="mb-6">
          <Notice tone="good">
            Signed attestation <span className="font-mono text-xs">{version.attestation_sha256.slice(0, 16)}…</span>{" "}
            binds this audit to the model artifact.
          </Notice>
        </div>
      ) : null}
      {audit ? (
        <div className="mb-6 space-y-4">
          <Card title="Entity audit">
            <p className="text-sm text-slate-600">
              {audit.entities_audited} personal-data entities tested with the{" "}
              <span className="font-mono text-xs">{audit.primary_method}</span> attack (chosen as the strongest by
              cross-fitting). Findings are controlled at a 5% false discovery rate; high and critical ones are
              re-tested independently. Values are masked.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {audit.taxonomy.map((t) => (
                <Badge key={t}>{t}</Badge>
              ))}
            </div>
            <div className="mt-4 flex flex-wrap gap-4 text-sm">
              {SEVERITIES.map((s) => (
                <span key={s} className="flex items-center gap-1.5">
                  <SeverityBadge severity={s} />
                  <span className="font-semibold">{audit.severity_counts[s] ?? 0}</span>
                </span>
              ))}
            </div>
          </Card>
          {audit.findings.length > PAGE ? (
            <p className="text-sm text-slate-500">
              {showAll ? `All ${audit.findings.length} findings, riskiest first. ` : `The ${PAGE} riskiest of ${audit.findings.length} findings. `}
              <TextLink href={showAll ? `/o/${org}/versions/${version.id}` : `/o/${org}/versions/${version.id}?show=all`}>
                {showAll ? "Show fewer" : "Show all"}
              </TextLink>
            </p>
          ) : null}
          <Table head={["Entity", "Type", "Severity", "Risk", "q-value", "Confirmed", "Record"]} empty="No significant findings.">
            {(showAll ? audit.findings : audit.findings.slice(0, PAGE)).map((f, i) => (
              <tr key={`${f.record}-${i}`}>
                <Td mono>{f.masked_value}</Td>
                <Td mono>{f.entity_type}</Td>
                <Td>
                  <SeverityBadge severity={f.severity} />
                </Td>
                <Td mono>{f.risk.toFixed(1)}</Td>
                <Td mono>{f.q_value.toFixed(3)}</Td>
                <Td>{f.confirmed === null ? "—" : f.confirmed ? "yes" : "no"}</Td>
                <Td mono>{f.record}</Td>
              </tr>
            ))}
          </Table>
        </div>
      ) : (
        <div className="mb-6">
          <Notice>No audit results to show yet.</Notice>
        </div>
      )}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <h2 className="mb-3 text-sm font-semibold text-slate-900">Canary test reports</h2>
          <Table head={["Submitted", "Attempts", "Leaks", "Summary"]} empty="No canary reports yet.">
            {reports.map((r) => (
              <tr key={r.id}>
                <Td>{formatDate(r.created_at)}</Td>
                <Td>{r.prompt_count}</Td>
                <Td>
                  <span className={r.leaks_found ? "font-semibold text-red-600" : ""}>{r.leaks_found}</span>
                </Td>
                <Td>{r.summary || "—"}</Td>
              </tr>
            ))}
          </Table>
        </div>
        {testing && can(role, "testreports.submit", me.super_admin) ? (
          <Card title="Report a canary test">
            <ActionForm action={submitTestReport} hidden={{ org, version: version.id }} label="Submit">
              <Field label="Extraction attempts">
                <input name="prompts" type="number" min={1} required className={inputClass} />
              </Field>
              <Field label="Attempts that surfaced real personal data">
                <input name="leaks" type="number" min={0} defaultValue={0} required className={inputClass} />
              </Field>
              <Field label="What you tried" hint="Describe techniques; never paste the leaked values.">
                <textarea name="summary" rows={3} maxLength={4000} className={inputClass} />
              </Field>
            </ActionForm>
          </Card>
        ) : null}
      </div>
    </>
  );
}
