import { ApiError, api } from "@/lib/api";

/** M9: the diligence report as a standalone page, fetched with the viewer's own session. */
export async function GET(_: Request, ctx: RouteContext<"/o/[org]/models/[id]/report">) {
  const { org, id } = await ctx.params;
  try {
    const html = await api<string>(`/models/${encodeURIComponent(id)}/diligence?format=html`, {
      org,
      raw: true,
    });
    return new Response(html, {
      headers: {
        "Content-Type": "text/html; charset=utf-8",
        "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:; sandbox",
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    if (error instanceof ApiError) return new Response(error.message, { status: error.status });
    throw error;
  }
}
