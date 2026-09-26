import type { Metadata } from "next";

import { ActionForm } from "@/components/ActionForm";
import { Field, inputClass } from "@/components/ui";

import { signIn } from "../auth";

export const metadata: Metadata = { title: "Sign in · Model Passport" };

export default function LoginPage() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-50 px-4">
      <div className="w-full max-w-sm">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-600 font-bold text-white">
            MP
          </div>
          <h1 className="text-xl font-semibold text-slate-900">Sign in to Model Passport</h1>
          <p className="mt-1 text-sm text-slate-500">Find and fix personal data your models memorized.</p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <ActionForm action={signIn} label="Sign in">
            <Field label="Email">
              <input name="email" type="email" required autoComplete="email" className={inputClass} />
            </Field>
            <Field label="Password">
              <input
                name="password"
                type="password"
                required
                autoComplete="current-password"
                className={inputClass}
              />
            </Field>
          </ActionForm>
        </div>
      </div>
    </main>
  );
}
