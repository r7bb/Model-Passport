import type { Metadata } from "next";

import { ActionForm } from "@/components/ActionForm";
import { Card, Field, Notice, PageHeader, Table, Td, inputClass } from "@/components/ui";
import { api } from "@/lib/api";
import { can } from "@/lib/platform";
import { load, requireMember } from "@/lib/session";
import type { GuardAction, GuardSettings } from "@/lib/types";

import { saveSettings } from "../actions";

export const metadata: Metadata = { title: "Guard settings" };

const LABELS: Record<GuardAction, string> = {
  mask: "Mask: hide it, put it back in the reply",
  redact: "Redact: remove it for good",
  block: "Block the whole message",
  allow: "Allow",
};
const REQUEST_ACTIONS: GuardAction[] = ["mask", "redact", "block", "allow"];
// In a reply there is nothing to put back, so masking is the same as redacting.
const REPLY_ACTIONS: GuardAction[] = ["redact", "block", "allow"];

function Choice({
  name,
  value,
  options,
  label,
}: {
  name: string;
  value: GuardAction;
  options: GuardAction[];
  label?: string;
}) {
  const current = options.includes(value) ? value : "redact";
  return (
    <select name={name} defaultValue={current} aria-label={label} className={inputClass}>
      {options.map((a) => (
        <option key={a} value={a}>
          {LABELS[a]}
        </option>
      ))}
    </select>
  );
}

function Fields({ settings }: { settings: GuardSettings }) {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 lg:grid-cols-3">
        <Field label="Provider URL" hint="Any OpenAI-compatible API">
          <input name="upstream_url" required defaultValue={settings.upstream_url} className={inputClass} />
        </Field>
        <Field label="Provider API key" hint="Stored encrypted and never shown again">
          <input
            name="upstream_key"
            type="password"
            autoComplete="off"
            placeholder={settings.has_upstream_key ? "Saved. Leave blank to keep it" : "sk-…"}
            className={inputClass}
          />
        </Field>
        <Field label="Default model" hint="Used when an app does not name one">
          <input name="default_model" required defaultValue={settings.default_model} className={inputClass} />
        </Field>
      </div>
      <Field
        label="Values the model memorized"
        hint="Exact values your entity audits found a model can repeat. Checked in every reply."
      >
        <Choice name="memorized" value={settings.policy.memorized} options={REPLY_ACTIONS} />
      </Field>
      <Table head={["Type of value", "In requests to the model", "In the model's replies"]}>
        {settings.entity_types.map((kind) => (
          <tr key={kind}>
            <Td mono>{kind}</Td>
            <Td>
              <Choice
                name={`in_${kind}`}
                value={settings.effective[kind].request} options={REQUEST_ACTIONS}
                label={`${kind} in requests`}
              />
            </Td>
            <Td>
              <Choice
                name={`out_${kind}`}
                value={settings.effective[kind].reply}
                options={REPLY_ACTIONS}
                label={`${kind} in replies`}
              />
            </Td>
          </tr>
        ))}
      </Table>
    </div>
  );
}

/** Which AI provider MP Guard forwards to, and what it does with each type of personal data. */
export default async function GuardSettingsPage({ params }: PageProps<"/o/[org]/guard/settings">) {
  const { org } = await params;
  const { me, role } = await requireMember(org);
  const manage = can(role, "guard.manage", me.super_admin);
  const settings = await load(api<GuardSettings>("/guard/settings", { org }));
  return (
    <>
      <PageHeader title="Guard settings" subtitle="Changes apply to the next request, for every app." />
      {manage ? (
        <Card>
          <ActionForm action={saveSettings} hidden={{ org }} label="Save settings">
            <Fields settings={settings} />
          </ActionForm>
        </Card>
      ) : (
        <>
          <div className="mb-4">
            <Notice>You can review these settings. Admins and ML engineers can change them.</Notice>
          </div>
          <Card>
            <fieldset disabled>
              <Fields settings={settings} />
            </fieldset>
          </Card>
        </>
      )}
    </>
  );
}
