import type { Metadata } from "next";

import { Badge, Card, Notice, PageHeader, Stat, Table, Td, TextLink } from "@/components/ui";
import { api } from "@/lib/api";
import { OUTCOME_TONE, describeReport, fillDays, quickstart } from "@/lib/guard";
import { formatDate } from "@/lib/platform";
import { load } from "@/lib/session";
import type { GuardEvent, GuardSettings, GuardStats } from "@/lib/types";

export const metadata: Metadata = { title: "Guard" };

const DAYS = 7;
const RECENT = 20;
// Where apps reach the API; it can differ from MP_API_URL, which is the server-to-server address.
const PUBLIC_API_URL = process.env.MP_PUBLIC_API_URL ?? process.env.MP_API_URL ?? "http://localhost:8080";

function DailyBars({ stats }: { stats: GuardStats }) {
  const days = fillDays(stats.per_day, DAYS);
  const peak = Math.max(1, ...days.map((d) => d.requests));
  const summary = days.map((d) => `${d.day}: ${d.requests}`).join(", ");
  return (
    <div className="flex h-32 items-end gap-2" role="img" aria-label={`Requests per day: ${summary}`}>
      {days.map((d) => (
        <div key={d.day} className="flex flex-1 flex-col items-center gap-1">
          <span className="text-[11px] text-slate-500">{d.requests || ""}</span>
          <div className="flex w-full flex-col-reverse overflow-hidden rounded-t bg-slate-100" style={{ height: "6rem" }}>
            <div className="bg-indigo-400" style={{ height: `${(100 * d.requests) / peak}%` }} title={`${d.day}: ${d.requests}`} />
          </div>
          <span className="text-[11px] text-slate-500">{d.day.slice(5)}</span>
        </div>
      ))}
    </div>
  );
}

function ByType({ counts }: { counts: Record<string, number> }) {
  const rows = Object.entries(counts);
  const peak = Math.max(1, ...rows.map(([, n]) => n));
  if (rows.length === 0) return <p className="text-sm text-slate-500">Nothing caught yet.</p>;
  return (
    <ul className="space-y-2">
      {rows.map(([type, n]) => (
        <li key={type} className="text-sm">
          <div className="flex justify-between">
            <span className="font-mono text-xs">{type}</span>
            <span className="font-semibold">{n}</span>
          </div>
          <div className="mt-1 h-1.5 rounded bg-slate-100">
            <div className="h-1.5 rounded bg-indigo-400" style={{ width: `${(100 * n) / peak}%` }} />
          </div>
        </li>
      ))}
    </ul>
  );
}

/** MP Guard at a glance: traffic, what was kept from the models, and how to connect an app. */
export default async function GuardPage({ params }: PageProps<"/o/[org]/guard">) {
  const { org } = await params;
  const [stats, settings, events] = await Promise.all([
    load(api<GuardStats>(`/guard/stats?days=${DAYS}`, { org })),
    load(api<GuardSettings>("/guard/settings", { org })),
    load(api<GuardEvent[]>(`/guard/events?limit=${RECENT}`, { org })),
  ]);
  const protectedCount = (stats.outcomes.protected ?? 0) + (stats.outcomes.blocked ?? 0);
  return (
    <>
      <PageHeader
        title="MP Guard"
        subtitle="Personal data is kept from the AI models your apps call, and checked again in every reply."
      />
      {settings.has_upstream_key ? null : (
        <div className="mb-6">
          <Notice tone="warn">
            No AI provider is connected yet. Add one in{" "}
            <TextLink href={`/o/${org}/guard/settings`}>Guard settings</TextLink>, then create a key for each app.
          </Notice>
        </div>
      )}
      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label={`Requests, ${DAYS} days`} value={stats.requests} />
        <Stat label="Protected or blocked" value={protectedCount} tone={protectedCount ? "good" : undefined} />
        <Stat label="Values kept from models" value={stats.values.masked + stats.values.redacted} />
        <Stat
          label="Memorized values caught"
          value={stats.values.memorized}
          tone={stats.values.memorized ? "bad" : undefined}
        />
      </div>
      <div className="mb-6 grid gap-6 lg:grid-cols-3">
        <Card title="Requests per day" className="lg:col-span-2">
          <DailyBars stats={stats} />
          <p className="mt-3 text-xs text-slate-500">
            Median added time: {stats.median_latency_ms === null ? "—" : `${stats.median_latency_ms} ms`}
          </p>
        </Card>
        <Card title="Kept from models, by type">
          <ByType counts={stats.by_type.masked} />
        </Card>
      </div>
      <Card title="Connect an app" className="mb-6">
        <p className="mb-3 text-sm text-slate-600">
          Apps keep their OpenAI SDK and change two settings. Create a key in{" "}
          <TextLink href={`/o/${org}/guard/keys`}>API keys</TextLink>.
        </p>
        <pre className="overflow-x-auto rounded-lg bg-slate-900 p-4 font-mono text-xs leading-relaxed text-slate-100">
          {quickstart(PUBLIC_API_URL, settings.endpoint)}
        </pre>
      </Card>
      <h2 className="mb-3 text-sm font-semibold text-slate-900">Recent activity</h2>
      <Table head={["When", "Source", "Model", "Outcome", "What the guard did", "Time"]} empty="No requests yet.">
        {events.map((e) => (
          <tr key={e.id}>
            <Td>{formatDate(e.created_at)}</Td>
            <Td>{e.source}</Td>
            <Td mono>{e.model || "—"}</Td>
            <Td>
              <Badge tone={OUTCOME_TONE[e.outcome]}>{e.outcome}</Badge>
            </Td>
            <Td>
              {describeReport(e.report)}
              {e.detail ? <span className="block text-xs text-slate-500">{e.detail}</span> : null}
            </Td>
            <Td mono>{e.latency_ms} ms</Td>
          </tr>
        ))}
      </Table>
    </>
  );
}
