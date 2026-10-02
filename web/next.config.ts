import type { NextConfig } from "next";

// Two years, subdomains included. No "preload": joining the browser preload list is hard to undo.
const HSTS = "max-age=63072000; includeSubDomains";

// Report-only for now, so a missed source shows up in the browser console instead of breaking a
// page; switch the header name to Content-Security-Policy once it reports nothing. Next.js
// inlines its bootstrap scripts and styles, hence 'unsafe-inline'. No report endpoint exists yet,
// so there is no report-uri / report-to.
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data:",
  "font-src 'self'",
  "connect-src 'self'",
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "object-src 'none'",
].join("; ");

const nextConfig: NextConfig = {
  output: "standalone",
  poweredByHeader: false,
  experimental: {
    serverActions: { bodySizeLimit: "50mb" }, // corpus uploads
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "same-origin" },
          { key: "Strict-Transport-Security", value: HSTS },
          { key: "Content-Security-Policy-Report-Only", value: CSP },
        ],
      },
    ];
  },
};

export default nextConfig;
