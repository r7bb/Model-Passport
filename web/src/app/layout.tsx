import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Model Passport", template: "%s · Model Passport" },
  description:
    "Model Passport finds the personal data a machine learning model has memorized, removes it, checks the model again, and records each step in a signed passport.",
  // A private console: keep every page out of search engines.
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full bg-slate-50 text-slate-900">{children}</body>
    </html>
  );
}
