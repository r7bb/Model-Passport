import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = { title: "Page not found" };

export default function NotFound() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-50 px-4">
      <div className="w-full max-w-sm text-center">
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-600 font-bold text-white">
          MP
        </div>
        <h1 className="text-xl font-semibold text-slate-900">Page not found</h1>
        <p className="mt-1 text-sm text-slate-600">This address doesn’t match any page. It may have moved, or the link may be mistyped.</p>
        <Link
          href="/"
          className="mt-6 inline-block rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
        >
          Go to your dashboard
        </Link>
      </div>
    </main>
  );
}
