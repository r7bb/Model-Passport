import { type ChainStatus, EventLog } from "@/components/EventLog";
import { PageHeader } from "@/components/ui";
import { api } from "@/lib/api";
import { load } from "@/lib/session";
import type { AuditEvent } from "@/lib/types";

export default async function PlatformLogPage() {
  const [events, status] = await Promise.all([
    load(api<AuditEvent[]>("/platform/audit-log")),
    load(api<ChainStatus>("/platform/audit-log/verify")),
  ]);
  return (
    <>
      <PageHeader title="Platform audit log" subtitle="Organizations created, plans changed, admins added." />
      <EventLog events={events} status={status} />
    </>
  );
}
