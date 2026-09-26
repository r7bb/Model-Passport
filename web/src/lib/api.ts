import "server-only";

import { cookies } from "next/headers";

/**
 * The backend client. It runs only on the server: the session token lives in an httpOnly
 * cookie that browser JavaScript can never read, and every call carries the organization.
 */

export const SESSION_COOKIE = "mp_session";
const API_URL = process.env.MP_API_URL ?? "http://localhost:8080";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

type Options = {
  method?: string;
  org?: string;
  json?: unknown;
  form?: FormData;
  token?: string;
  raw?: boolean;
};

async function detail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) return body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    // not JSON
  }
  return `${response.status} ${response.statusText}`;
}

export async function api<T = unknown>(path: string, options: Options = {}): Promise<T> {
  const token = options.token ?? (await cookies()).get(SESSION_COOKIE)?.value;
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (options.org) headers["X-MP-Tenant"] = options.org;
  let body: BodyInit | undefined;
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.json);
  } else if (options.form) {
    body = options.form;
  }
  const response = await fetch(`${API_URL}/api/v1${path}`, {
    method: options.method ?? (body ? "POST" : "GET"),
    headers,
    body,
    cache: "no-store",
  });
  if (!response.ok) throw new ApiError(response.status, await detail(response));
  if (options.raw) return (await response.text()) as T;
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}
