"use server";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";

import { type ActionState, attempt, field } from "@/lib/actions";
import { SESSION_COOKIE, api } from "@/lib/api";

type Token = { access_token: string; expires_in_minutes: number };

export async function signIn(_: ActionState, form: FormData): Promise<ActionState> {
  const result = await attempt(async () => {
    const token = await api<Token>("/auth/login", {
      json: { email: field(form, "email"), password: form.get("password") ?? "" },
      token: "",
    });
    (await cookies()).set(SESSION_COOKIE, token.access_token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      maxAge: token.expires_in_minutes * 60,
    });
    return "Signed in";
  });
  if (result?.ok) redirect("/");
  return result;
}

export async function signOut(): Promise<void> {
  (await cookies()).delete(SESSION_COOKIE);
  redirect("/login");
}
