import type { Metadata } from "next";

import { ActionForm } from "@/components/ActionForm";
import { Badge, Card, Field, PageHeader, Table, Td, TextLink, inputClass } from "@/components/ui";
import { api } from "@/lib/api";
import { can, formatDate } from "@/lib/platform";
import { load, requireMember } from "@/lib/session";
import type { Dataset, Model } from "@/lib/types";

import { uploadDataset } from "../actions";

export const metadata: Metadata = { title: "Data" };

function size(bytes: number): string {
  return bytes < 1024 ? `${bytes} B` : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`;
}

/** M9: where training data came from, under what license and consent, stored encrypted. */
export default async function DataPage({ params }: PageProps<"/o/[org]/data">) {
  const { org } = await params;
  const { me, role } = await requireMember(org);
  const allow = (p: Parameters<typeof can>[1]) => can(role, p, me.super_admin);
  const [datasets, models] = await Promise.all([
    load(api<Dataset[]>("/datasets", { org })),
    allow("reports.read") && allow("models.read") ? load(api<Model[]>("/models", { org })) : Promise.resolve([]),
  ]);
  return (
    <>
      <PageHeader title="Data provenance" subtitle="Every corpus is fingerprinted and encrypted with this organization's own key." />
      <div className={`grid gap-6 ${allow("models.write") ? "lg:grid-cols-3" : ""}`}>
        <div className="space-y-6 lg:col-span-2">
          <Table head={["Dataset", "Source", "License", "Consent", "Records", "Entities", "Size", "SHA-256"]} empty="No datasets yet.">
            {datasets.map((d) => (
              <tr key={d.id}>
                <Td>
                  <div className="font-medium text-slate-900">{d.name}</div>
                  <div className="text-xs text-slate-400">{formatDate(d.created_at)}</div>
                </Td>
                <Td>{d.source || "—"}</Td>
                <Td>{d.license}</Td>
                <Td>
                  <Badge tone={d.consent === "unknown" ? "warn" : "good"}>{d.consent}</Badge>
                </Td>
                <Td>{d.records}</Td>
                <Td>{d.entities}</Td>
                <Td>{size(d.size_bytes)}</Td>
                <Td mono>{d.sha256.slice(0, 12)}</Td>
              </tr>
            ))}
          </Table>
          {models.length ? (
            <Card title="Diligence reports">
              <p className="mb-3 text-sm text-slate-500">
                For investors and acquirers: data sources, every audit, signed attestations, and the audit-log check.
              </p>
              <ul className="space-y-1 text-sm">
                {models.map((m) => (
                  <li key={m.id}>
                    <TextLink href={`/o/${org}/models/${m.id}/report`}>{m.name}</TextLink>
                  </li>
                ))}
              </ul>
            </Card>
          ) : null}
        </div>
        {allow("models.write") ? (
          <Card title="Upload a corpus">
            <ActionForm action={uploadDataset} hidden={{ org }} label="Upload">
              <Field label="File" hint="JSONL: one record per line with text and entities.">
                <input name="file" type="file" required accept=".jsonl,.json,.txt" className="block w-full text-sm" />
              </Field>
              <Field label="Name">
                <input name="name" required className={inputClass} />
              </Field>
              <Field label="Source">
                <input name="source" className={inputClass} placeholder="support tickets 2025" />
              </Field>
              <Field label="License">
                <input name="license" defaultValue="unknown" className={inputClass} />
              </Field>
              <Field label="Consent">
                <select name="consent" defaultValue="unknown" className={inputClass}>
                  <option value="unknown">unknown</option>
                  <option value="obtained">obtained</option>
                  <option value="contract">contract</option>
                  <option value="legitimate-interest">legitimate interest</option>
                  <option value="public">public</option>
                </select>
              </Field>
            </ActionForm>
          </Card>
        ) : null}
      </div>
    </>
  );
}
