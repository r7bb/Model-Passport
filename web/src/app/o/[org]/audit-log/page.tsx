import { type ChainStatus, EventLog } from "@/components/EventLog";
import { PageHeader } from "@/components/ui";
import { api } from "@/lib/api";
import { load } from "@/lib/session";
import type { AuditEvent } from "@/lib/types";

/** M4: the organization's tamper-evident record of every action. */
export default async function AuditLogPage({ params }: PageProps<"/o/[org]/audit-log">) {
  const { org } = await params;
  const [events, status] = await Promise.all([
    load(api<AuditEvent[]>("/audit-log?limit=1000", { org })),
    load(api<ChainStatus>("/audit-log/verify", { org })),
  ]);
  return (
    <>
      <PageHeader title="Audit log" subtitle="Everything anyone (or any worker) did in this organization." />
      <EventLog events={events} status={status} />
    </>
  );
}
