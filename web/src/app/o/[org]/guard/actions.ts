"use server";

import { revalidatePath } from "next/cache";

import { type ActionState, attempt, field, idOf, slugOf } from "@/lib/actions";
import { ApiError, api } from "@/lib/api";
import type { GuardAction, GuardReport, GuardSettings } from "@/lib/types";

/** MP Guard actions, called with the signed-in person's own session (the backend decides). */

const ACTIONS: GuardAction[] = ["mask", "redact", "block", "allow"];

async function inOrg(
  form: FormData,
  run: (org: string) => Promise<string | { message: string; secret: string }>,
): Promise<ActionState> {
  return attempt(async () => {
    const org = slugOf(form);
    const result = await run(org);
    revalidatePath(`/o/${org}/guard`, "layout");
    return result;
  });
}

function action(form: FormData, name: string, fallback: GuardAction): GuardAction {
  const value = field(form, name) as GuardAction;
  return ACTIONS.includes(value) ? value : fallback;
}

export async function saveSettings(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    const current = await api<GuardSettings>("/guard/settings", { org });
    const inbound: Record<string, GuardAction> = {};
    const outbound: Record<string, GuardAction> = {};
    for (const kind of current.entity_types) {
      inbound[kind] = action(form, `in_${kind}`, current.effective[kind].request);
      outbound[kind] = action(form, `out_${kind}`, current.effective[kind].reply);
    }
    const key = form.get("upstream_key");
    await api("/guard/settings", {
      org,
      method: "PUT",
      json: {
        upstream_url: field(form, "upstream_url"),
        // Blank keeps the saved key: it is never sent back to the browser.
        upstream_key: typeof key === "string" && key.trim() ? key.trim() : null,
        default_model: field(form, "default_model"),
        policy: {
          inbound,
          inbound_default: current.policy.inbound_default,
          outbound,
          memorized: action(form, "memorized", current.policy.memorized),
        },
      },
    });
    return "Saved";
  });
}

export async function createKey(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    const created = await api<{ key: string; name: string }>("/guard/keys", {
      org,
      json: { name: field(form, "name") },
    });
    return { message: `Created ${created.name}`, secret: created.key };
  });
}

export async function revokeKey(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    await api(`/guard/keys/${idOf(form, "key")}`, { org, method: "DELETE" });
    return "Revoked: apps using it are refused from now on";
  });
}

export type PlaygroundState = {
  ok: boolean;
  // What was submitted: React resets the form after an action, back to these.
  text?: string;
  model?: string;
  message?: string;
  blocked?: boolean;
  sees?: string | null;
  reply?: string | null;
  report?: GuardReport;
} | null;

type PlaygroundReply = {
  blocked: boolean;
  message?: string;
  model_sees?: string | null;
  reply?: string | null;
  report: GuardReport;
};

export async function tryGuard(_: PlaygroundState, form: FormData): Promise<PlaygroundState> {
  const text = String(form.get("text") ?? "");
  const model = field(form, "model");
  try {
    const org = slugOf(form);
    const out = await api<PlaygroundReply>("/guard/playground", {
      org,
      json: { text, send: form.get("send") === "yes", model },
    });
    revalidatePath(`/o/${org}/guard`, "layout");
    return {
      ok: !out.blocked && !out.message,
      text,
      model,
      message: out.message,
      blocked: out.blocked,
      sees: out.model_sees,
      reply: out.reply,
      report: out.report,
    };
  } catch (error) {
    if (error instanceof ApiError) return { ok: false, message: error.message, text, model };
    throw error;
  }
}
