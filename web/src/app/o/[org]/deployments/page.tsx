import { Badge, Notice, PageHeader, Table, Td } from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import { formatDate } from "@/lib/platform";
import { load } from "@/lib/session";
import type { Deployment } from "@/lib/types";

/** M8: what serves where. Developer endpoints for canary testing; consumer endpoints once released. */
export default async function DeploymentsPage({ params }: PageProps<"/o/[org]/deployments">) {
  const { org } = await params;
  let deployments: Deployment[] = [];
  let unavailable = "";
  try {
    deployments = await load(api<Deployment[]>("/deployments", { org }));
  } catch (error) {
    if (!(error instanceof ApiError) || error.status < 500) throw error;
    unavailable = error.message;
  }
  return (
    <>
      <PageHeader
        title="Dev endpoints & testing"
        subtitle="Consumers can reach only approved, released versions; the kill switch stops all of them."
      />
      {unavailable ? (
        <Notice tone="warn">Deployments are managed by the control plane: {unavailable}</Notice>
      ) : (
        <Table head={["Model", "Version", "Environment", "Status", "Endpoint", "Since"]} empty="Nothing deployed yet.">
          {deployments.map((d) => (
            <tr key={d.id}>
              <Td>{d.model}</Td>
              <Td mono>{d.version}</Td>
              <Td>
                <Badge tone={d.environment === "consumer" ? "good" : "info"}>{d.environment}</Badge>
              </Td>
              <Td>
                <Badge tone={d.status === "active" ? "good" : d.status === "killed" ? "bad" : "neutral"}>{d.status}</Badge>
              </Td>
              <Td mono>{d.endpoint}</Td>
              <Td>{formatDate(d.created_at)}</Td>
            </tr>
          ))}
        </Table>
      )}
    </>
  );
}
