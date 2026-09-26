"use server";

import { revalidatePath } from "next/cache";

import { type ActionState, attempt, field, idOf, slugOf } from "@/lib/actions";
import { ApiError, api } from "@/lib/api";

/**
 * Organization actions. Each one names the organization and calls the backend with the
 * signed-in person's own session, so the backend's role checks and lifecycle rules decide.
 */

async function inOrg(form: FormData, run: (org: string) => Promise<string>): Promise<ActionState> {
  return attempt(async () => {
    const org = slugOf(form);
    const message = await run(org);
    revalidatePath(`/o/${org}`, "layout");
    return message;
  });
}

// M2 Access & roles

export async function addMember(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    const password = form.get("password");
    await api("/members", {
      org,
      json: {
        email: field(form, "email"),
        name: field(form, "name"),
        role: field(form, "role"),
        password: typeof password === "string" && password ? password : null,
      },
    });
    return `Added ${field(form, "email")}`;
  });
}

export async function changeRole(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    await api(`/members/${idOf(form, "user")}`, { org, method: "PATCH", json: { role: field(form, "role") } });
    return "Role changed";
  });
}

export async function removeMember(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    await api(`/members/${idOf(form, "user")}`, { org, method: "DELETE" });
    return "Removed";
  });
}

// M5 Models and M6 versions

export async function registerModel(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    await api("/models", {
      org,
      json: {
        name: field(form, "name"),
        access: field(form, "access"),
        base: field(form, "base"),
        description: field(form, "description"),
      },
    });
    return `Registered ${field(form, "name")}`;
  });
}

export async function registerVersion(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    const dataset = field(form, "dataset");
    await api(`/models/${idOf(form, "model")}/versions`, {
      org,
      json: {
        version: field(form, "version") || "1.0.0",
        dataset_id: dataset ? idOf(form, "dataset") : null,
        reference_dataset_id: idOf(form, "reference"),
        train: form.get("train") === "on",
      },
    });
    return form.get("train") === "on" ? "Registered; training and audit queued" : "Registered";
  });
}

// Lifecycle: detect -> remediate -> verify & deploy

export async function startAudit(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    await api(`/versions/${idOf(form, "version")}/audit`, { org, method: "POST" });
    return "Audit queued";
  });
}

export async function startRemediation(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    await api(`/versions/${idOf(form, "version")}/remediate`, { org, method: "POST" });
    return "Remediation queued: sanitize, retrain, re-audit";
  });
}

export async function transition(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    const to = field(form, "to");
    await api(`/versions/${idOf(form, "version")}/transition`, {
      org,
      json: { to, reason: field(form, "reason") },
    });
    return `Moved to ${to.replace("_", " ")}`;
  });
}

export async function decide(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    const decision = field(form, "decision") === "approve" ? "approve" : "reject";
    await api(`/versions/${idOf(form, "version")}/approvals`, {
      org,
      json: { decision, comment: field(form, "comment") },
    });
    return decision === "approve" ? "Approved" : "Rejected";
  });
}

export async function kill(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    await api(`/versions/${idOf(form, "version")}/kill`, { org, json: { reason: field(form, "reason") } });
    return "Kill switch activated: blocked everywhere";
  });
}

export async function rollback(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    const restored = await api<{ version: string }>(`/models/${idOf(form, "model")}/rollback`, {
      org,
      method: "POST",
    });
    return `Rolled back to ${restored.version}`;
  });
}

export async function generateReport(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    await api(`/models/${idOf(form, "model")}/reports`, { org, method: "POST" });
    return "Report generation queued";
  });
}

// M8 Canary testing

export async function submitTestReport(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    await api(`/versions/${idOf(form, "version")}/test-reports`, {
      org,
      json: {
        prompt_count: Number(field(form, "prompts")),
        leaks_found: Number(field(form, "leaks")),
        summary: field(form, "summary"),
      },
    });
    return "Report submitted";
  });
}

// M9 Data provenance

export async function uploadDataset(_: ActionState, form: FormData): Promise<ActionState> {
  return inOrg(form, async (org) => {
    const file = form.get("file");
    if (!(file instanceof File) || file.size === 0) throw new ApiError(400, "choose a corpus file first");
    const upload = new FormData();
    upload.set("file", file, file.name);
    for (const name of ["name", "source", "license", "consent"]) upload.set(name, field(form, name));
    await api("/datasets", { org, form: upload });
    return `Uploaded ${field(form, "name")}`;
  });
}
