import { ActionForm } from "@/components/ActionForm";
import { inputClass } from "@/components/ui";
import { type Role, type State, nextSteps } from "@/lib/platform";

import { decide, kill, startAudit, startRemediation, transition } from "@/app/o/[org]/actions";

/** The lifecycle steps this person can take on a version right now. */
export function Lifecycle({
  org,
  versionId,
  state,
  role,
  superAdmin,
}: {
  org: string;
  versionId: string;
  state: State;
  role: Role | null;
  superAdmin: boolean;
}) {
  const steps = nextSteps(state, role, superAdmin);
  if (steps.length === 0) return <span className="text-xs text-slate-400">No steps for your role</span>;
  const hidden = { org, version: versionId };
  return (
    <div className="flex flex-wrap items-start gap-2">
      {steps.map((step) => {
        switch (step.kind) {
          case "audit":
            return <ActionForm key={step.label} action={startAudit} hidden={hidden} label={step.label} inline />;
          case "remediate":
            return <ActionForm key={step.label} action={startRemediation} hidden={hidden} label={step.label} inline />;
          case "approve":
          case "reject":
            return (
              <ActionForm
                key={step.label}
                action={decide}
                hidden={{ ...hidden, decision: step.kind }}
                label={step.label}
                danger={step.danger}
                confirm={step.kind === "reject" ? "Reject this version?" : undefined}
                inline
              />
            );
          case "kill":
            return (
              <ActionForm
                key={step.label}
                action={kill}
                hidden={hidden}
                label={step.label}
                danger
                confirm="Block this version everywhere, immediately?"
                inline
              >
                <input name="reason" required minLength={3} placeholder="Reason" className={`${inputClass} w-40`} />
              </ActionForm>
            );
          default:
            return (
              <ActionForm
                key={step.label}
                action={transition}
                hidden={{ ...hidden, to: step.to ?? "" }}
                label={step.label}
                danger={step.danger}
                inline
              />
            );
        }
      })}
    </div>
  );
}
