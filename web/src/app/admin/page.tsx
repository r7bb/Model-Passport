import { ActionForm } from "@/components/ActionForm";
import { Badge, Card, Field, PageHeader, Table, Td, inputClass } from "@/components/ui";
import { api } from "@/lib/api";
import { formatDate } from "@/lib/platform";
import { load } from "@/lib/session";
import type { Tenant } from "@/lib/types";

import { addOrgAdmin, createTenant, updateTenant } from "./actions";

const PLANS = ["free", "team", "enterprise"] as const;

function PlanSelect({ name, value }: { name: string; value?: string }) {
  return (
    <select name={name} defaultValue={value ?? "free"} className={inputClass}>
      {PLANS.map((p) => (
        <option key={p} value={p}>
          {p}
        </option>
      ))}
    </select>
  );
}

export default async function AdminPage() {
  const tenants = await load(api<Tenant[]>("/platform/tenants"));
  const base = process.env.MP_BASE_DOMAIN;
  return (
    <>
      <PageHeader
        title="Organizations"
        subtitle="Every organization is isolated: its own data, keys, audit log, and address."
      />
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <Table head={["Organization", "Address", "Plan", "Status", "Created", ""]} empty="No organizations yet.">
            {tenants.map((t) => (
              <tr key={t.id}>
                <Td>
                  <div className="font-medium text-slate-900">{t.name}</div>
                  <div className="text-xs text-slate-400">{t.slug}</div>
                </Td>
                <Td mono>{base ? `${t.slug}.${base}` : `/o/${t.slug}`}</Td>
                <Td>
                  <ActionForm action={updateTenant} hidden={{ slug: t.slug }} label="Set" inline>
                    <div className="w-28">
                      <PlanSelect name="plan" value={t.plan} />
                    </div>
                  </ActionForm>
                </Td>
                <Td>
                  <Badge tone={t.status === "active" ? "good" : "bad"}>{t.status}</Badge>
                </Td>
                <Td>{formatDate(t.created_at)}</Td>
                <Td>
                  <ActionForm
                    action={updateTenant}
                    hidden={{ slug: t.slug, status: t.status === "active" ? "suspended" : "active" }}
                    label={t.status === "active" ? "Suspend" : "Reactivate"}
                    danger={t.status === "active"}
                    confirm={t.status === "active" ? `Suspend ${t.name}? Its members lose access.` : undefined}
                    inline
                  />
                </Td>
              </tr>
            ))}
          </Table>
        </div>
        <div className="space-y-6">
          <Card title="New organization">
            <ActionForm action={createTenant} label="Create">
              <Field label="Name">
                <input name="name" required className={inputClass} placeholder="Acme Health" />
              </Field>
              <Field label="Address" hint="Lowercase letters, digits, and hyphens.">
                <input name="slug" required pattern="[a-z0-9][a-z0-9-]*[a-z0-9]" className={inputClass} placeholder="acme" />
              </Field>
              <Field label="Plan">
                <PlanSelect name="plan" />
              </Field>
            </ActionForm>
          </Card>
          <Card title="Add an org admin">
            <ActionForm action={addOrgAdmin} label="Add admin">
              <Field label="Organization">
                <select name="slug" required className={inputClass}>
                  {tenants.map((t) => (
                    <option key={t.id} value={t.slug}>
                      {t.name}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Email">
                <input name="email" type="email" required className={inputClass} />
              </Field>
              <Field label="Name">
                <input name="name" className={inputClass} />
              </Field>
              <Field label="Initial password" hint="Only for people without an account yet (12+ characters).">
                <input name="password" type="password" minLength={12} autoComplete="new-password" className={inputClass} />
              </Field>
            </ActionForm>
          </Card>
        </div>
      </div>
    </>
  );
}
