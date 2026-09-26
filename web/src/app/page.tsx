import { redirect } from "next/navigation";

import { requireMe } from "@/lib/session";

/** Send each person to where they work: the platform console or their organization. */
export default async function Home() {
  const me = await requireMe();
  if (me.super_admin) redirect("/admin");
  const first = me.memberships[0];
  if (first) redirect(`/o/${first.tenant}`);
  return (
    <main className="mx-auto max-w-md px-4 py-24 text-center">
      <h1 className="text-xl font-semibold">No organization yet</h1>
      <p className="mt-2 text-sm text-slate-500">
        Ask your organization admin to add {me.email} to their workspace.
      </p>
    </main>
  );
}
