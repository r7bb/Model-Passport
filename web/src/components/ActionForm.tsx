"use client";

import { type ReactNode, useActionState, useState } from "react";
import { useFormStatus } from "react-dom";

type ActionState = { ok: boolean; message: string; secret?: string } | null;
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

/** A secret shown once (a new API key), with a copy button. */
function Secret({ value }: { value: string }) {
  const [status, setStatus] = useState("Copy");
  return (
    <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3">
      <p className="text-xs font-medium text-amber-800">Copy it now: it will not be shown again.</p>
      <div className="mt-2 flex items-center gap-2">
        <code className="min-w-0 flex-1 truncate rounded bg-white px-2 py-1 font-mono text-xs text-slate-800 ring-1 ring-amber-200">
          {value}
        </code>
        <button
          type="button"
          onClick={() => {
            navigator.clipboard.writeText(value).then(
              () => setStatus("Copied"),
              () => setStatus("Select it and copy"),
            );
          }}
          className="rounded-md bg-white px-2 py-1 text-xs font-medium text-amber-800 ring-1 ring-amber-300 hover:bg-amber-100"
        >
          {status}
        </button>
      </div>
    </div>
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
      {state?.secret ? <Secret key={state.secret} value={state.secret} /> : null}
    </form>
  );
}
