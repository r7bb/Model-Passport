import type { Metadata } from "next";

import { ActionForm } from "@/components/ActionForm";
import { Badge, Card, Field, PageHeader, Table, Td, inputClass } from "@/components/ui";
import { api } from "@/lib/api";
import { can, formatDate } from "@/lib/platform";
import { load, requireMember } from "@/lib/session";
import type { GuardKey } from "@/lib/types";

import { createKey, revokeKey } from "../actions";

export const metadata: Metadata = { title: "Guard API keys" };

/** API keys your apps use to call MP Guard. */
export default async function KeysPage({ params }: PageProps<"/o/[org]/guard/keys">) {
  const { org } = await params;
  const { me, role } = await requireMember(org);
  const manage = can(role, "guard.manage", me.super_admin);
  const keys = await load(api<GuardKey[]>("/guard/keys", { org }));
  return (
    <>
      <PageHeader
        title="API keys"
        subtitle="One key per app, so you can see each app's traffic and revoke it on its own."
      />
      <div className={`grid gap-6 ${manage ? "lg:grid-cols-3" : ""}`}>
        <div className="lg:col-span-2">
          <Table head={["App", "Key", "Created", "Last used", "Status", ""]} empty="No keys yet.">
            {keys.map((k) => (
              <tr key={k.id}>
                <Td>
                  <span className="font-medium text-slate-900">{k.name}</span>
                </Td>
                <Td mono>{k.hint}…</Td>
                <Td>{formatDate(k.created_at)}</Td>
                <Td>{k.last_used_at ? formatDate(k.last_used_at) : "never"}</Td>
                <Td>
                  <Badge tone={k.revoked_at ? "neutral" : "good"}>{k.revoked_at ? "revoked" : "active"}</Badge>
                </Td>
                <Td>
                  {manage && !k.revoked_at ? (
                    <ActionForm
                      action={revokeKey}
                      hidden={{ org, key: k.id }}
                      label="Revoke"
                      danger
                      confirm={`Revoke ${k.name}? Apps using it will be refused immediately.`}
                      inline
                    />
                  ) : null}
                </Td>
              </tr>
            ))}
          </Table>
        </div>
        {manage ? (
          <Card title="New key">
            <ActionForm action={createKey} hidden={{ org }} label="Create key">
              <Field label="App name" hint="For example: support-bot, intake-agent, staging">
                <input name="name" required maxLength={200} className={inputClass} />
              </Field>
            </ActionForm>
          </Card>
        ) : null}
      </div>
    </>
  );
}
