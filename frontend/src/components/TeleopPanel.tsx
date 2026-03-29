import { Button } from "@/components/ui/button.tsx";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card.tsx";
import type { RobotStateName } from "@/lib/types.ts";

interface TeleopPanelProps {
  state: RobotStateName;
  inflight: boolean;
  onStartTeleop: () => void;
  onStopTeleop: () => void;
}

export function TeleopPanel({
  state,
  inflight,
  onStartTeleop,
  onStopTeleop,
}: TeleopPanelProps) {
  const idle = state === "idle";

  return (
    <Card className="flex-1">
      <CardHeader>
        <CardTitle>Remote Teleop</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-1.5">
          <Button
            size="sm"
            disabled={inflight || !idle}
            onClick={onStartTeleop}
          >
            Start
          </Button>
          <Button
            variant="destructive"
            size="sm"
            disabled={inflight || state !== "teleop"}
            onClick={onStopTeleop}
          >
            Stop
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
