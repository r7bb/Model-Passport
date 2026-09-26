import { AutoRefresh } from "@/components/AutoRefresh";
import { Badge, PageHeader, Table, Td, type Tone } from "@/components/ui";
import { api } from "@/lib/api";
import { formatDate } from "@/lib/platform";
import { load } from "@/lib/session";
import type { Job } from "@/lib/types";

const TONE: Record<Job["status"], Tone> = {
  queued: "neutral",
  running: "info",
  succeeded: "good",
  failed: "bad",
  cancelled: "warn",
};

/** M7: training, audit, remediation, and report jobs run by the workers. */
export default async function JobsPage({ params }: PageProps<"/o/[org]/jobs">) {
  const { org } = await params;
  const jobs = await load(api<Job[]>("/jobs?limit=100", { org }));
  return (
    <>
      <AutoRefresh active={jobs.some((j) => j.status === "queued" || j.status === "running")} />
      <PageHeader title="Pipelines & workers" subtitle="Updates live while jobs are running." />
      <Table head={["Job", "Status", "Attempts", "Queued", "Finished", "Outcome"]} empty="No jobs yet.">
        {jobs.map((j) => (
          <tr key={j.id}>
            <Td mono>{j.kind}</Td>
            <Td>
              <Badge tone={TONE[j.status]}>{j.status}</Badge>
            </Td>
            <Td>{j.attempts}</Td>
            <Td>{formatDate(j.created_at)}</Td>
            <Td>{formatDate(j.finished_at)}</Td>
            <Td>
              {j.error ? (
                <span className="text-red-600">{j.error}</span>
              ) : j.result ? (
                <span className="font-mono text-xs text-slate-500">
                  {Object.entries(j.result)
                    .filter(([, v]) => typeof v !== "object")
                    .slice(0, 3)
                    .map(([k, v]) => `${k}=${String(v)}`)
                    .join(" ")}
                </span>
              ) : (
                "—"
              )}
            </Td>
          </tr>
        ))}
      </Table>
    </>
  );
}
