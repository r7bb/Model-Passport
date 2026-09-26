import Link from "next/link";

import { Shell } from "@/components/Shell";
import { ROLE_LABELS, visibleModules } from "@/lib/platform";
import { requireMember } from "@/lib/session";

export default async function OrgLayout({ children, params }: LayoutProps<"/o/[org]">) {
  const { org } = await params;
  const { me, role } = await requireMember(org);
  const nav = visibleModules(role, me.super_admin).map((m) => ({
    href: `/o/${org}${m.path}`,
    label: m.label,
    tag: m.id,
  }));
  if (me.super_admin) nav.push({ href: "/admin", label: "Platform console", tag: "" });
  const others = me.memberships.filter((m) => m.tenant !== org);
  const switcher = others.length ? (
    <details className="border-b border-slate-200 px-5 py-2 text-xs">
      <summary className="cursor-pointer text-slate-500">Switch organization</summary>
      <div className="mt-2 space-y-1">
        {others.map((m) => (
          <Link key={m.tenant} href={`/o/${m.tenant}`} className="block text-indigo-600 hover:text-indigo-500">
            {m.tenant}
          </Link>
        ))}
      </div>
    </details>
  ) : undefined;
  return (
    <Shell
      title={org}
      subtitle={role ? ROLE_LABELS[role] : "Super admin"}
      nav={nav}
      who={me.email}
      switcher={switcher}
    >
      {children}
    </Shell>
  );
}
