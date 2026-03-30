import { useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button.tsx";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card.tsx";
import { Select } from "@/components/ui/select.tsx";
import type { RobotStateName } from "@/lib/types.ts";

interface InferencePanelProps {
  state: RobotStateName;
  inflight: boolean;
  models: Record<string, string>;
  onStartInference: (model: string) => void;
  onStopInference: (model: string) => void;
}

export function InferencePanel({
  state,
  inflight,
  models,
  onStartInference,
  onStopInference,
}: InferencePanelProps) {
  const [selected, setSelected] = useState("");
  const idle = state === "idle";

  return (
    <Card className="flex-1">
      <CardHeader>
        <CardTitle>Policy Inference</CardTitle>
      </CardHeader>
      <CardContent>
        <Select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="mb-1.5"
        >
          <option value="">Select model...</option>
          {Object.entries(models).map(([key, name]) => (
            <option key={key} value={key}>
              {name}
            </option>
          ))}
        </Select>
        <div className="flex flex-wrap gap-1.5">
          <Button
            size="sm"
            disabled={inflight || !idle || !selected}
            onClick={() => onStartInference(selected)}
          >
            {inflight && idle && (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            )}
            Start
          </Button>
          <Button
            variant="destructive"
            size="sm"
            disabled={inflight || state !== "inferring"}
            onClick={() => onStopInference(selected)}
          >
            Stop
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
