import { Badge, Card, PageHeader, Stat, Table, Td } from "@/components/ui";
import { api } from "@/lib/api";
import { formatAuc, formatDate } from "@/lib/platform";
import { load } from "@/lib/session";

type Analytics = {
  risk_over_time: {
    model: string;
    version: string;
    auc: number | null;
    critical: number;
    high: number;
    verdict: string | null;
    at: string;
  }[];
  high_risk_findings_by_type: Record<string, number>;
  remediation_rounds: number;
};

/** M3: how leakage risk changes across versions, and which kinds of data leak most. */
export default async function AnalyticsPage({ params }: PageProps<"/o/[org]/analytics">) {
  const { org } = await params;
  const a = await load(api<Analytics>("/analytics", { org }));
  const types = Object.entries(a.high_risk_findings_by_type);
  const peak = Math.max(1, ...types.map(([, n]) => n));
  const latest = a.risk_over_time.at(-1);
  return (
    <>
      <PageHeader
        title="Analytics"
        subtitle="Attack AUC near 0.5 means an attacker cannot tell trained people from strangers."
      />
      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-3">
        <Stat label="Audited versions" value={a.risk_over_time.length} />
        <Stat label="Remediation rounds" value={a.remediation_rounds} />
        <Stat label="Latest attack AUC" value={formatAuc(latest?.auc)} />
      </div>
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <Table head={["Model", "Version", "Attack AUC", "Critical", "High", "Verdict", "Audited"]} empty="No audits yet.">
            {a.risk_over_time.map((r) => (
              <tr key={`${r.model}-${r.version}`}>
                <Td>{r.model}</Td>
                <Td mono>{r.version}</Td>
                <Td mono>{formatAuc(r.auc)}</Td>
                <Td>{r.critical}</Td>
                <Td>{r.high}</Td>
                <Td>
                  <Badge tone={r.verdict === "fail" ? "bad" : r.verdict === "pass" ? "good" : "warn"}>
                    {r.verdict ?? "—"}
                  </Badge>
                </Td>
                <Td>{formatDate(r.at)}</Td>
              </tr>
            ))}
          </Table>
        </div>
        <Card title="High-risk findings by data type">
          {types.length === 0 ? (
            <p className="text-sm text-slate-400">None so far.</p>
          ) : (
            <ul className="space-y-2">
              {types.map(([type, n]) => (
                <li key={type} className="text-sm">
                  <div className="flex justify-between">
                    <span className="font-mono text-xs">{type}</span>
                    <span className="font-semibold">{n}</span>
                  </div>
                  <div className="mt-1 h-1.5 rounded bg-slate-100">
                    <div className="h-1.5 rounded bg-red-400" style={{ width: `${(100 * n) / peak}%` }} />
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </>
  );
}
