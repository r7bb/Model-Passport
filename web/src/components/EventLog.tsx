import { Badge, Card, Table, Td } from "@/components/ui";
import { formatDate } from "@/lib/platform";
import type { AuditEvent } from "@/lib/types";

export type ChainStatus = { events: number; intact: boolean; problems: string[] };

/** A hash-chained audit log and whether its chain still verifies. */
export function EventLog({ events, status }: { events: AuditEvent[]; status: ChainStatus }) {
  return (
    <div className="space-y-4">
      <Card>
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <Badge tone={status.intact ? "good" : "bad"}>{status.intact ? "chain intact" : "chain broken"}</Badge>
          <span className="text-slate-600">
            {status.events} events, each sealed with the hash of the one before it, so edits or
            deletions are detectable.
          </span>
        </div>
        {status.problems.length ? (
          <ul className="mt-3 list-disc pl-5 text-sm text-red-600">
            {status.problems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        ) : null}
      </Card>
      <Table head={["#", "When", "Who", "Action", "Target", "Hash"]} empty="No events yet.">
        {[...events].reverse().map((e) => (
          <tr key={e.seq}>
            <Td>{e.seq}</Td>
            <Td>{formatDate(e.created_at)}</Td>
            <Td>{e.actor}</Td>
            <Td mono>{e.action}</Td>
            <Td mono>{e.target_type ? `${e.target_type}:${e.target_id.slice(0, 8)}` : "—"}</Td>
            <Td mono>{e.hash.slice(0, 12)}</Td>
          </tr>
        ))}
      </Table>
    </div>
  );
}
