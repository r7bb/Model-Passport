import Link from "next/link";
import type { ReactNode } from "react";

import { signOut } from "@/app/auth";

export type NavItem = { href: string; label: string; tag?: string };

/** The frame of every signed-in page: sidebar navigation and the signed-in person. */
export function Shell({
  title,
  subtitle,
  nav,
  who,
  switcher,
  children,
}: {
  title: string;
  subtitle: string;
  nav: NavItem[];
  who: string;
  switcher?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-64 shrink-0 flex-col border-r border-slate-200 bg-white md:flex">
        <div className="flex items-center gap-3 border-b border-slate-200 px-5 py-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-indigo-600 text-xs font-bold text-white">
            MP
          </div>
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold">{title}</div>
            <div className="truncate text-xs text-slate-500">{subtitle}</div>
          </div>
        </div>
        {switcher}
        <nav className="flex-1 space-y-0.5 px-3 py-3">
          {nav.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="flex items-center justify-between rounded-md px-3 py-2 text-sm text-slate-700 hover:bg-slate-100"
            >
              <span>{item.label}</span>
              {item.tag ? <span className="text-[10px] font-medium text-slate-400">{item.tag}</span> : null}
            </Link>
          ))}
        </nav>
        <div className="border-t border-slate-200 px-5 py-4">
          <div className="truncate text-xs text-slate-500">{who}</div>
          <form action={signOut}>
            <button type="submit" className="mt-1 text-xs font-medium text-indigo-600 hover:text-indigo-500">
              Sign out
            </button>
          </form>
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3 md:hidden">
          <span className="text-sm font-semibold">{title}</span>
          <details className="relative">
            <summary className="cursor-pointer text-sm text-indigo-600">Menu</summary>
            <nav className="absolute right-0 z-10 mt-2 w-56 rounded-md border border-slate-200 bg-white p-2 shadow-lg">
              {nav.map((item) => (
                <Link key={item.href} href={item.href} className="block rounded px-3 py-2 text-sm hover:bg-slate-100">
                  {item.label}
                </Link>
              ))}
            </nav>
          </details>
        </header>
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 md:px-8">{children}</main>
      </div>
    </div>
  );
}
