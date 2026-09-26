"use client";

import { type ReactNode, useActionState } from "react";
import { useFormStatus } from "react-dom";

type ActionState = { ok: boolean; message: string } | null;
type Action = (state: ActionState, form: FormData) => Promise<ActionState>;

function Submit({ label, danger }: { label: string; danger?: boolean }) {
  const { pending } = useFormStatus();
  const color = danger
    ? "bg-red-600 hover:bg-red-500 focus-visible:outline-red-600"
    : "bg-indigo-600 hover:bg-indigo-500 focus-visible:outline-indigo-600";
  return (
    <button
      type="submit"
      disabled={pending}
      className={`rounded-md px-3 py-1.5 text-sm font-semibold text-white shadow-sm focus-visible:outline-2 focus-visible:outline-offset-2 disabled:opacity-60 ${color}`}
    >
      {pending ? "Working…" : label}
    </button>
  );
}

/** A form bound to a server action, showing its result inline. */
export function ActionForm({
  action,
  hidden = {},
  label,
  danger = false,
  confirm,
  inline = false,
  children,
}: {
  action: Action;
  hidden?: Record<string, string>;
  label: string;
  danger?: boolean;
  confirm?: string;
  inline?: boolean;
  children?: ReactNode;
}) {
  const [state, formAction] = useActionState(action, null);
  return (
    <form
      action={formAction}
      onSubmit={(event) => {
        if (confirm && !window.confirm(confirm)) event.preventDefault();
      }}
      className={inline ? "inline-flex flex-wrap items-center gap-2" : "space-y-3"}
    >
      {Object.entries(hidden).map(([name, value]) => (
        <input key={name} type="hidden" name={name} value={value} />
      ))}
      {children}
      <div className="flex flex-wrap items-center gap-3">
        <Submit label={label} danger={danger} />
        {state ? (
          <span role="status" className={`text-sm ${state.ok ? "text-emerald-600" : "text-red-600"}`}>
            {state.message}
          </span>
        ) : null}
      </div>
    </form>
  );
}
