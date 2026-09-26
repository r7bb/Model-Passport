import { ActionForm } from "@/components/ActionForm";
import { Badge, Card, Field, PageHeader, Table, Td, TextLink, inputClass } from "@/components/ui";
import { api } from "@/lib/api";
import { can, formatDate } from "@/lib/platform";
import { load, requireMember } from "@/lib/session";
import type { Model } from "@/lib/types";

import { registerModel } from "../actions";

/** M5: the organization's models, open-weight or behind an API. */
export default async function ModelsPage({ params }: PageProps<"/o/[org]/models">) {
  const { org } = await params;
  const { me, role } = await requireMember(org);
  const models = await load(api<Model[]>("/models", { org }));
  const canWrite = can(role, "models.write", me.super_admin);
  return (
    <>
      <PageHeader title="Models" subtitle="Each model keeps every version, audit, and decision." />
      <div className={`grid gap-6 ${canWrite ? "lg:grid-cols-3" : ""}`}>
        <div className="lg:col-span-2">
          <Table head={["Model", "Access", "Base", "Registered"]} empty="No models yet.">
            {models.map((m) => (
              <tr key={m.id}>
                <Td>
                  <TextLink href={`/o/${org}/models/${m.id}`}>{m.name}</TextLink>
                  {m.description ? <div className="text-xs text-slate-400">{m.description}</div> : null}
                </Td>
                <Td>
                  <Badge tone={m.access === "api" ? "info" : "neutral"}>{m.access}</Badge>
                </Td>
                <Td mono>{m.base}</Td>
                <Td>{formatDate(m.created_at)}</Td>
              </tr>
            ))}
          </Table>
        </div>
        {canWrite ? (
          <Card title="Register a model">
            <ActionForm action={registerModel} hidden={{ org }} label="Register">
              <Field label="Name">
                <input name="name" required className={inputClass} placeholder="support-assistant" />
              </Field>
              <Field label="Access">
                <select name="access" className={inputClass} defaultValue="open-weight">
                  <option value="open-weight">Open weights (likelihood attacks)</option>
                  <option value="api">API only (extraction probing)</option>
                </select>
              </Field>
              <Field label="Base" hint="hf:<id>, anthropic:<model>, openai:<model>, or tiny">
                <input name="base" required className={inputClass} placeholder="hf:distilgpt2" />
              </Field>
              <Field label="Description">
                <input name="description" className={inputClass} />
              </Field>
            </ActionForm>
          </Card>
        ) : null}
      </div>
    </>
  );
}
