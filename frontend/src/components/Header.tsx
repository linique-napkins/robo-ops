import { Badge } from "@/components/ui/badge.tsx";
import type { RobotStateName } from "@/lib/types.ts";

const STATE_BADGE: Record<
  RobotStateName,
  { label: string; variant: "zinc" | "blue" | "emerald" | "purple" | "amber" | "yellow" }
> = {
  disconnected: { label: "Disconnected", variant: "zinc" },
  idle: { label: "Idle", variant: "blue" },
  replaying: { label: "Replaying", variant: "emerald" },
  inferring: { label: "Inferring", variant: "purple" },
  teleop: { label: "Teleop", variant: "amber" },
  stowing: { label: "Stowing", variant: "yellow" },
};

interface HeaderProps {
  state: RobotStateName;
  paused: boolean;
  operation: string | null;
}

export function Header({ state, paused, operation }: HeaderProps) {
  const badge = STATE_BADGE[state] ?? STATE_BADGE.disconnected;
  let label = badge.label;
  if (operation) label = `${badge.label} (${operation})`;
  if (paused) label += " [PAUSED]";

  return (
    <header className="flex h-12 shrink-0 items-center justify-between border-b px-5">
      <h1 className="text-base font-semibold tracking-tight">SO101 Demo</h1>
      <Badge variant={badge.variant}>{label}</Badge>
    </header>
  );
}
