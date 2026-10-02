import type { Metadata } from "next";

import { Playground } from "@/components/Playground";
import { Notice, PageHeader, TextLink } from "@/components/ui";
import { api } from "@/lib/api";
import { can } from "@/lib/platform";
import { load, requireMember } from "@/lib/session";
import type { GuardSettings } from "@/lib/types";

export const metadata: Metadata = { title: "Guard playground" };

/** Try MP Guard on a message before connecting an app. */
export default async function PlaygroundPage({ params }: PageProps<"/o/[org]/guard/playground">) {
  const { org } = await params;
  const { me, role } = await requireMember(org);
  if (!can(role, "guard.manage", me.super_admin)) {
    return (
      <>
        <PageHeader title="Playground" />
        <Notice>Admins and ML engineers can try the guard here.</Notice>
      </>
    );
  }
  const settings = await load(api<GuardSettings>("/guard/settings", { org }));
  return (
    <>
      <PageHeader
        title="Playground"
        subtitle="See exactly what an AI model would receive, with your current policy."
      />
      {settings.has_upstream_key ? null : (
        <div className="mb-6">
          <Notice>
            Previews work now. To send messages to a model, connect your AI provider in{" "}
            <TextLink href={`/o/${org}/guard/settings`}>Guard settings</TextLink>.
          </Notice>
        </div>
      )}
      <Playground org={org} canSend={settings.has_upstream_key} model={settings.default_model} />
    </>
  );
}
