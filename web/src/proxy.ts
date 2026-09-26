import { type NextRequest, NextResponse } from "next/server";

import { tenantFromHost } from "@/lib/platform";

/**
 * Each organization has its own address, "<slug>.<MP_BASE_DOMAIN>". Requests there are served
 * by the /o/<slug> pages. Signed-out visitors go to sign-in (an optimistic check; every page
 * and action is still authorized by the backend).
 */

const SESSION_COOKIE = "mp_session";
const OPEN = ["/login"];

export function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;
  if (!OPEN.includes(pathname) && !request.cookies.has(SESSION_COOKIE)) {
    return NextResponse.redirect(new URL("/login", request.url));
  }
  const slug = tenantFromHost(request.headers.get("host"), process.env.MP_BASE_DOMAIN ?? "");
  if (slug && !pathname.startsWith("/o/") && !OPEN.includes(pathname)) {
    const url = request.nextUrl.clone();
    url.pathname = `/o/${slug}${pathname === "/" ? "" : pathname}`;
    return NextResponse.rewrite(url);
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|ico)$).*)"],
};
