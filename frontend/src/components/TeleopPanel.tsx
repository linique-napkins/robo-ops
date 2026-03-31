import { useState } from "react";
import { Button } from "@/components/ui/button.tsx";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card.tsx";
import { Select } from "@/components/ui/select.tsx";
import type { RobotStateName } from "@/lib/types.ts";

interface TeleopPanelProps {
  state: RobotStateName;
  inflight: boolean;
  onStartTeleop: (mode: string) => void;
  onStopTeleop: () => void;
}

export function TeleopPanel({
  state,
  inflight,
  onStartTeleop,
  onStopTeleop,
}: TeleopPanelProps) {
  const [mode, setMode] = useState("local");
  const idle = state === "idle";

  return (
    <Card className="flex-1">
      <CardHeader>
        <CardTitle>Teleop</CardTitle>
      </CardHeader>
      <CardContent>
        <Select
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          className="mb-1.5"
        >
          <option value="local">Local (Leader Arms)</option>
          <option value="remote">Remote (WebSocket)</option>
        </Select>
        <div className="flex flex-wrap gap-1.5">
          <Button
            size="sm"
            disabled={inflight || !idle}
            onClick={() => onStartTeleop(mode)}
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
