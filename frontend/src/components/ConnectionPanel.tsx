import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button.tsx";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card.tsx";
import type { RobotStateName } from "@/lib/types.ts";

interface ConnectionPanelProps {
  state: RobotStateName;
  paused: boolean;
  inflight: boolean;
  onConnect: () => void;
  onDisconnect: () => void;
  onStow: () => void;
  onTogglePause: () => void;
}

export function ConnectionPanel({
  state,
  paused,
  inflight,
  onConnect,
  onDisconnect,
  onStow,
  onTogglePause,
}: ConnectionPanelProps) {
  const connected = state !== "disconnected";
  const active = ["replaying", "inferring", "teleop"].includes(state);

  return (
    <Card className="flex-1">
      <CardHeader>
        <CardTitle>Connection</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-1.5">
          <Button
            variant="success"
            size="sm"
            disabled={inflight || connected}
            onClick={onConnect}
          >
            {inflight && state === "disconnected" && (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            )}
            Connect
          </Button>
          <Button
            variant="destructive"
            size="sm"
            disabled={inflight || !connected}
            onClick={onDisconnect}
          >
            Disconnect
          </Button>
          <Button
            variant="warning"
            size="sm"
            disabled={inflight || !connected || state === "stowing"}
            onClick={onStow}
          >
            {state === "stowing" && (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            )}
            Stow
          </Button>
          <Button
            size="sm"
            disabled={inflight || !active}
            onClick={onTogglePause}
          >
            {paused ? "Resume" : "Pause"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
