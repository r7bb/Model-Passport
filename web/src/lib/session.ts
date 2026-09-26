import "server-only";

import { redirect } from "next/navigation";
import { cache } from "react";

import { ApiError, api } from "./api";
import type { Role } from "./platform";
import type { Me } from "./types";

/** The signed-in person, fetched once per request; null when signed out or expired. */
export const getMe = cache(async (): Promise<Me | null> => {
  try {
    return await api<Me>("/me");
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) return null;
    throw error;
  }
});

export async function requireMe(): Promise<Me> {
  const me = await getMe();
  if (!me) redirect("/login");
  return me;
}

/** The person's role in an organization (null for super admins who are not members). */
export async function requireMember(org: string): Promise<{ me: Me; role: Role | null }> {
  const me = await requireMe();
  const membership = me.memberships.find((m) => m.tenant === org);
  if (!membership && !me.super_admin) redirect("/");
  return { me, role: membership?.role ?? null };
}

/** Run a backend call from a page: a lapsed session goes back to sign-in. */
export async function load<T>(call: Promise<T>): Promise<T> {
  try {
    return await call;
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) redirect("/login");
    throw error;
  }
}
