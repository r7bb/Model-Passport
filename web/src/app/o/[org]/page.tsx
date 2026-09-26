import { AutoRefresh } from "@/components/AutoRefresh";
import { Card, Notice, PageHeader, Stat, StateBadge, Table, Td, TextLink } from "@/components/ui";
import { api } from "@/lib/api";
import { type State, can, formatDate } from "@/lib/platform";
import { load, requireMember } from "@/lib/session";

type Dashboard = {
  organization: { slug: string; name: string; plan: string };
  models: number;
  versions_by_state: Partial<Record<State, number>>;
  open_findings: { critical: number; high: number };
  blocked: string[];
  jobs_in_progress: number;
  recent_activity: { seq: number; actor: string; action: string; at: string }[];
};

const PHASES: { title: string; states: State[] }[] = [
  { title: "1 · Detect", states: ["registered", "auditing", "findings", "clean"] },
  { title: "2 · Remediate", states: ["remediating", "superseded"] },
  { title: "3 · Verify & deploy", states: ["canary", "verifying", "approved", "released"] },
  { title: "Stopped", states: ["rolled_back", "rejected", "killed"] },
];

/** M1: where every model stands in the detect → remediate → verify & deploy flow. */
export default async function DashboardPage({ params }: PageProps<"/o/[org]">) {
  const { org } = await params;
  const { me, role } = await requireMember(org);
  if (!can(role, "models.read", me.super_admin)) {
    return (
      <>
        <PageHeader title="Welcome" />
        <Notice>
          {role === "external_reviewer"
            ? "You can open the diligence reports shared with you under Data provenance."
            : "Your role uses approved models through their consumer endpoints; there is nothing to manage here."}
        </Notice>
      </>
    );
  }
  const d = await load(api<Dashboard>("/dashboard", { org }));
  const count = (s: State) => d.versions_by_state[s] ?? 0;
  const risky = d.open_findings.critical + d.open_findings.high;
  return (
    <>
      <AutoRefresh active={d.jobs_in_progress > 0} />
      <PageHeader
        title={d.organization.name}
        subtitle={`${d.organization.plan} plan · personal-data leakage across your models`}
        actions={<TextLink href={`/o/${org}/models`}>View models →</TextLink>}
      />
      {d.blocked.length ? (
        <div className="mb-6">
          <Notice tone="bad">
            {d.blocked.length} version{d.blocked.length > 1 ? "s are" : " is"} blocked by the kill switch.
          </Notice>
        </div>
      ) : null}
      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Models" value={d.models} />
        <Stat label="Open high-risk findings" value={risky} tone={risky ? "bad" : "good"} />
        <Stat label="Live for consumers" value={count("released")} tone={count("released") ? "good" : undefined} />
        <Stat label="Jobs running" value={d.jobs_in_progress} />
      </div>
      <div className="mb-6 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {PHASES.map((phase) => (
          <Card key={phase.title} title={phase.title}>
            <ul className="space-y-2">
              {phase.states.map((s) => (
                <li key={s} className="flex items-center justify-between text-sm">
                  <StateBadge state={s} />
                  <span className="font-semibold text-slate-700">{count(s)}</span>
                </li>
              ))}
            </ul>
          </Card>
        ))}
      </div>
      <h2 className="mb-3 text-sm font-semibold text-slate-900">Recent activity</h2>
      <Table head={["When", "Who", "What"]} empty="No activity yet.">
        {d.recent_activity.map((e) => (
          <tr key={e.seq}>
            <Td>{formatDate(e.at)}</Td>
            <Td>{e.actor}</Td>
            <Td mono>{e.action}</Td>
          </tr>
        ))}
      </Table>
    </>
  );
}
