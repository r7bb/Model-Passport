"use server";

import { revalidatePath } from "next/cache";

import { type ActionState, attempt, field, slugOf } from "@/lib/actions";
import { api } from "@/lib/api";

/** Super-admin actions; the backend refuses them for anyone else. */

export async function createTenant(_: ActionState, form: FormData): Promise<ActionState> {
  return attempt(async () => {
    const slug = slugOf(form, "slug");
    await api("/platform/tenants", {
      json: { slug, name: field(form, "name"), plan: field(form, "plan") || "free" },
    });
    revalidatePath("/admin");
    return `Created ${slug}`;
  });
}

export async function updateTenant(_: ActionState, form: FormData): Promise<ActionState> {
  return attempt(async () => {
    const slug = slugOf(form, "slug");
    const body: Record<string, string> = {};
    if (field(form, "status")) body.status = field(form, "status");
    if (field(form, "plan")) body.plan = field(form, "plan");
    await api(`/platform/tenants/${slug}`, { method: "PATCH", json: body });
    revalidatePath("/admin");
    return "Saved";
  });
}

export async function addOrgAdmin(_: ActionState, form: FormData): Promise<ActionState> {
  return attempt(async () => {
    const slug = slugOf(form, "slug");
    const password = form.get("password");
    await api(`/platform/tenants/${slug}/admins`, {
      json: {
        email: field(form, "email"),
        name: field(form, "name"),
        role: "org_admin",
        password: typeof password === "string" && password ? password : null,
      },
    });
    revalidatePath("/admin");
    return `Added ${field(form, "email")} as org admin`;
  });
}
