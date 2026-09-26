import "server-only";

import { ApiError } from "./api";

/** What a form shows after its server action ran. */
export type ActionState = { ok: boolean; message: string } | null;

export function field(form: FormData, name: string): string {
  const value = form.get(name);
  return typeof value === "string" ? value.trim() : "";
}

const SLUG = /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/;
const ID = /^[A-Za-z0-9_-]{1,64}$/;

/** Validate identifiers from a form before they reach a backend URL. */
export function slugOf(form: FormData, name = "org"): string {
  const value = field(form, name);
  if (!SLUG.test(value)) throw new ApiError(400, "invalid organization");
  return value;
}

export function idOf(form: FormData, name: string): string {
  const value = field(form, name);
  if (!ID.test(value)) throw new ApiError(400, `invalid ${name}`);
  return value;
}

/** Run a mutation and turn backend refusals into a message for the form. */
export async function attempt(run: () => Promise<string>): Promise<ActionState> {
  try {
    return { ok: true, message: await run() };
  } catch (error) {
    if (error instanceof ApiError) {
      const prefix = error.status === 403 ? "Not allowed: " : error.status === 401 ? "Signed out: " : "";
      return { ok: false, message: prefix + error.message };
    }
    throw error;
  }
}
