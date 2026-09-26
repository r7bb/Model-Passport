import { ActionForm } from "@/components/ActionForm";
import { Card, Field, PageHeader, Table, Td, inputClass } from "@/components/ui";
import { api } from "@/lib/api";
import { ROLE_LABELS, type Role } from "@/lib/platform";
import { load, requireMember } from "@/lib/session";
import type { Member } from "@/lib/types";

import { addMember, changeRole, removeMember } from "../actions";

const ROLES = Object.keys(ROLE_LABELS) as Role[];

const DUTIES: Record<Role, string> = {
  org_admin: "Manages people, approvals, and releases",
  ml_engineer: "Registers, audits, and retrains models",
  compliance_auditor: "Reviews findings, approves releases, can pull the kill switch",
  canary_tester: "Tries to extract personal data from developer endpoints",
  end_consumer: "Uses approved models",
  external_reviewer: "Reads diligence reports (investors, acquirers)",
};

function RoleSelect({ value }: { value?: Role }) {
  return (
    <select name="role" defaultValue={value ?? "ml_engineer"} className={inputClass}>
      {ROLES.map((r) => (
        <option key={r} value={r}>
          {ROLE_LABELS[r]}
        </option>
      ))}
    </select>
  );
}

/** M2: who belongs to the organization and what each person may do. */
export default async function MembersPage({ params }: PageProps<"/o/[org]/members">) {
  const { org } = await params;
  const { me } = await requireMember(org);
  const members = await load(api<Member[]>("/members", { org }));
  return (
    <>
      <PageHeader title="Access & roles" subtitle="Each person has one role in this organization." />
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <Table head={["Person", "Role", ""]}>
            {members.map((m) => (
              <tr key={m.user_id}>
                <Td>
                  <div className="font-medium text-slate-900">{m.name || m.email}</div>
                  <div className="text-xs text-slate-400">{m.email}</div>
                </Td>
                <Td>
                  <ActionForm action={changeRole} hidden={{ org, user: m.user_id }} label="Change" inline>
                    <div className="w-48">
                      <RoleSelect value={m.role} />
                    </div>
                  </ActionForm>
                </Td>
                <Td>
                  {m.user_id === me.id ? (
                    <span className="text-xs text-slate-400">you</span>
                  ) : (
                    <ActionForm
                      action={removeMember}
                      hidden={{ org, user: m.user_id }}
                      label="Remove"
                      danger
                      confirm={`Remove ${m.email} from ${org}?`}
                      inline
                    />
                  )}
                </Td>
              </tr>
            ))}
          </Table>
        </div>
        <div className="space-y-6">
          <Card title="Add a person">
            <ActionForm action={addMember} hidden={{ org }} label="Add">
              <Field label="Email">
                <input name="email" type="email" required className={inputClass} />
              </Field>
              <Field label="Name">
                <input name="name" className={inputClass} />
              </Field>
              <Field label="Role">
                <RoleSelect />
              </Field>
              <Field label="Initial password" hint="Only for people without an account yet (12+ characters).">
                <input name="password" type="password" minLength={12} autoComplete="new-password" className={inputClass} />
              </Field>
            </ActionForm>
          </Card>
          <Card title="Roles">
            <dl className="space-y-2 text-sm">
              {ROLES.map((r) => (
                <div key={r}>
                  <dt className="font-medium text-slate-800">{ROLE_LABELS[r]}</dt>
                  <dd className="text-slate-500">{DUTIES[r]}</dd>
                </div>
              ))}
            </dl>
          </Card>
        </div>
      </div>
    </>
  );
}
