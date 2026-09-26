import { redirect } from "next/navigation";

import { Shell } from "@/components/Shell";
import { requireMe } from "@/lib/session";

export default async function AdminLayout({ children }: LayoutProps<"/admin">) {
  const me = await requireMe();
  if (!me.super_admin) redirect("/");
  return (
    <Shell
      title="Model Passport"
      subtitle="Platform console"
      who={me.email}
      nav={[
        { href: "/admin", label: "Organizations" },
        { href: "/admin/audit-log", label: "Platform audit log" },
        ...me.memberships.map((m) => ({ href: `/o/${m.tenant}`, label: m.tenant, tag: "org" })),
      ]}
    >
      {children}
    </Shell>
  );
}
