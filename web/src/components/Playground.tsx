"use client";

import { useActionState } from "react";
import { useFormStatus } from "react-dom";

import { type PlaygroundState, tryGuard } from "@/app/o/[org]/guard/actions";

const EXAMPLE =
  "Hi, I'm Dana Lee (dana.lee@example.com, 617-555-0142). Please move my refund to card 4111 1111 1111 1111 and confirm by email.";

function Buttons({ canSend }: { canSend: boolean }) {
  const { pending } = useFormStatus();
  const base = "rounded-md px-3 py-1.5 text-sm font-semibold shadow-sm disabled:opacity-60";
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="submit"
        name="send"
        value="no"
        disabled={pending}
        className={`${base} bg-white text-slate-700 ring-1 ring-slate-300 hover:bg-slate-50`}
      >
        {pending ? "Working…" : "Preview what the model sees"}
      </button>
      <button
        type="submit"
        name="send"
        value="yes"
        disabled={pending || !canSend}
        title={canSend ? undefined : "Connect an AI provider in Guard settings first"}
        className={`${base} bg-indigo-600 text-white hover:bg-indigo-500`}
      >
        Send to the model
      </button>
    </div>
  );
}

function Counts({ report }: { report: NonNullable<PlaygroundState>["report"] }) {
  const rows = [
    ["Kept from the model", report?.masked],
    ["Redacted in the reply", report?.redacted],
    ["Produced by the model", report?.leaked],
    ["Memorized values caught", report?.memorized],
  ] as const;
  return (
    <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
      {rows.map(([label, counts]) => (
        <div key={label} className="rounded-lg bg-slate-50 p-3">
          <dt className="text-xs text-slate-500">{label}</dt>
          <dd className="mt-1 font-semibold text-slate-900">
            {Object.values(counts ?? {}).reduce((a, b) => a + b, 0)}
          </dd>
          <dd className="mt-0.5 truncate font-mono text-[11px] text-slate-400">
            {Object.keys(counts ?? {}).join(", ") || "—"}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function Panel({ title, children, tone = "plain" }: { title: string; children: React.ReactNode; tone?: "plain" | "model" }) {
  const color = tone === "model" ? "border-indigo-200 bg-indigo-50/40" : "border-slate-200 bg-white";
  return (
    <div className={`rounded-xl border p-4 ${color}`}>
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</div>
      <div className="whitespace-pre-wrap text-sm text-slate-800">{children}</div>
    </div>
  );
}

/** Try the guard on any text: what the model would see, and what comes back. */
export function Playground({ org, canSend, model }: { org: string; canSend: boolean; model: string }) {
  const [state, formAction] = useActionState<PlaygroundState, FormData>(tryGuard, null);
  return (
    <div className="space-y-6">
      <form action={formAction} className="space-y-3 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <input type="hidden" name="org" value={org} />
        <label className="block">
          <span className="mb-1 block text-xs font-medium text-slate-600">A message, as one of your users would write it</span>
          <textarea
            name="text"
            rows={4}
            required
            defaultValue={state?.text ?? EXAMPLE}
            className="block w-full rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
        </label>
        <label className="block max-w-xs">
          <span className="mb-1 block text-xs font-medium text-slate-600">Model (blank for the default)</span>
          <input
            name="model"
            defaultValue={state?.model ?? ""}
            placeholder={model || "gpt-4o-mini"}
            className="block w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm shadow-sm"
          />
        </label>
        <Buttons canSend={canSend} />
      </form>
      {state ? (
        <div className="space-y-4">
          {state.message ? (
            <div
              role="status"
              className={`rounded-lg px-4 py-3 text-sm ring-1 ring-inset ${state.blocked ? "bg-red-50 text-red-700 ring-red-200" : "bg-amber-50 text-amber-800 ring-amber-200"}`}
            >
              {state.blocked ? "Blocked by your policy: " : ""}
              {state.message}
            </div>
          ) : null}
          {state.sees ? (
            <div className="grid gap-4 lg:grid-cols-2">
              <Panel title="What the AI model receives" tone="model">
                {state.sees}
              </Panel>
              <Panel title="What your user gets back">
                {state.reply ?? "Preview only. Send it to see the model's reply, with values restored."}
              </Panel>
            </div>
          ) : null}
          {state.report ? <Counts report={state.report} /> : null}
        </div>
      ) : null}
    </div>
  );
}
