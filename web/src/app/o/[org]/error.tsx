"use client";

import { Notice } from "@/components/ui";

export default function OrgError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <div className="space-y-4">
      <Notice tone="bad">
        This page could not be loaded. Your role may not include it, or the service is unavailable.
        {error.digest ? <span className="ml-1 text-xs opacity-70">(ref {error.digest})</span> : null}
      </Notice>
      <button type="button" onClick={reset} className="text-sm font-medium text-indigo-600 hover:text-indigo-500">
        Try again
      </button>
    </div>
  );
}
