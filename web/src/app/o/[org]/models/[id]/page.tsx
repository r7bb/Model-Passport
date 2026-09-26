import { ActionForm } from "@/components/ActionForm";
import { AutoRefresh } from "@/components/AutoRefresh";
import { Lifecycle } from "@/components/Lifecycle";
import { Badge, Card, Field, PageHeader, StateBadge, Table, Td, TextLink, inputClass } from "@/components/ui";
import { api } from "@/lib/api";
import { can, formatAuc, formatDate, nextVersion } from "@/lib/platform";
import { load, requireMember } from "@/lib/session";
import type { Dataset, ModelDetail } from "@/lib/types";

import { generateReport, registerVersion, rollback } from "../../actions";

const BUSY = new Set(["auditing", "remediating"]);

/** M6: a model's versions, how each was derived, and the next lifecycle step for each. */
export default async function ModelPage({ params }: PageProps<"/o/[org]/models/[id]">) {
  const { org, id } = await params;
  const { me, role } = await requireMember(org);
  const allow = (p: Parameters<typeof can>[1]) => can(role, p, me.super_admin);
  const model = await load(api<ModelDetail>(`/models/${encodeURIComponent(id)}`, { org }));
  const datasets = allow("models.write") ? await load(api<Dataset[]>("/datasets", { org })) : [];
  const versions = [...model.versions].reverse();
  const byId = new Map(model.versions.map((v) => [v.id, v.version]));
  return (
    <>
      <AutoRefresh active={model.versions.some((v) => BUSY.has(v.state))} />
      <PageHeader
        title={model.name}
        subtitle={
          <>
            <Badge tone={model.access === "api" ? "info" : "neutral"}>{model.access}</Badge>{" "}
            <span className="font-mono text-xs">{model.base}</span>
          </>
        }
        actions={
          <>
            {allow("reports.read") ? (
              <>
                <a
                  href={`/o/${org}/models/${model.id}/report`}
                  target="_blank"
                  rel="noopener"
                  className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
                >
                  Diligence report
                </a>
                <ActionForm action={generateReport} hidden={{ org, model: model.id }} label="Archive report" inline />
              </>
            ) : null}
            {allow("releases.manage") ? (
              <ActionForm
                action={rollback}
                hidden={{ org, model: model.id }}
                label="Roll back release"
                danger
                confirm="Retire the live version and restore the previous release?"
                inline
              />
            ) : null}
          </>
        }
      />
      <div className="space-y-6">
        <Table head={["Version", "State", "Attack AUC", "Critical / high", "From", "Updated", "Next step"]} empty="No versions yet.">
          {versions.map((v) => (
            <tr key={v.id}>
              <Td>
                <TextLink href={`/o/${org}/versions/${v.id}`}>{v.version}</TextLink>
              </Td>
              <Td>
                <StateBadge state={v.state} />
              </Td>
              <Td mono>{formatAuc(v.auc)}</Td>
              <Td>
                <span className={v.critical + v.high ? "font-semibold text-red-600" : ""}>
                  {v.critical} / {v.high}
                </span>
              </Td>
              <Td mono>{v.parent_id ? (byId.get(v.parent_id) ?? "—") : "—"}</Td>
              <Td>{formatDate(v.updated_at)}</Td>
              <Td>
                <Lifecycle org={org} versionId={v.id} state={v.state} role={role} superAdmin={me.super_admin} />
              </Td>
            </tr>
          ))}
        </Table>
        {allow("models.write") ? (
          <Card title="Register a version">
            {datasets.length === 0 ? (
              <p className="text-sm text-slate-500">
                Upload the training corpus under <TextLink href={`/o/${org}/data`}>Data provenance</TextLink> first.
              </p>
            ) : (
              <ActionForm action={registerVersion} hidden={{ org, model: model.id }} label="Register version">
                <div className="grid gap-3 md:grid-cols-3">
                  <Field label="Version">
                    <input name="version" defaultValue={nextVersion(model.versions.map((v) => v.version))} className={inputClass} />
                  </Field>
                  <Field label="Audit against" hint="The original corpus with its sensitive entities.">
                    <select name="reference" required className={inputClass}>
                      {datasets.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Trained on">
                    <select name="dataset" className={inputClass} defaultValue="">
                      <option value="">Not recorded</option>
                      {datasets.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name}
                        </option>
                      ))}
                    </select>
                  </Field>
                </div>
                {model.access === "open-weight" ? (
                  <label className="flex items-center gap-2 text-sm text-slate-600">
                    <input type="checkbox" name="train" /> Fine-tune on the platform, then audit
                  </label>
                ) : null}
              </ActionForm>
            )}
          </Card>
        ) : null}
      </div>
    </>
  );
}
