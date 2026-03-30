import { useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button.tsx";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card.tsx";
import { Select } from "@/components/ui/select.tsx";
import type { RobotStateName } from "@/lib/types.ts";

interface ReplayPanelProps {
  state: RobotStateName;
  inflight: boolean;
  recordings: string[];
  onStartReplay: (name: string, loop: boolean) => void;
  onStopReplay: (name: string) => void;
}

export function ReplayPanel({
  state,
  inflight,
  recordings,
  onStartReplay,
  onStopReplay,
}: ReplayPanelProps) {
  const [selected, setSelected] = useState("");
  const idle = state === "idle";

  return (
    <Card className="flex-1">
      <CardHeader>
        <CardTitle>Demo Replay</CardTitle>
      </CardHeader>
      <CardContent>
        <Select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="mb-1.5"
        >
          <option value="">Select recording...</option>
          {recordings.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </Select>
        <div className="flex flex-wrap gap-1.5">
          <Button
            size="sm"
            disabled={inflight || !idle || !selected}
            onClick={() => onStartReplay(selected, false)}
          >
            {inflight && state === "idle" && (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            )}
            Play
          </Button>
          <Button
            size="sm"
            disabled={inflight || !idle || !selected}
            onClick={() => onStartReplay(selected, true)}
          >
            Loop
          </Button>
          <Button
            variant="destructive"
            size="sm"
            disabled={inflight || state !== "replaying"}
            onClick={() => onStopReplay(selected)}
          >
            Stop
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
