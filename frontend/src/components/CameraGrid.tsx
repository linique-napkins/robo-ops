import { useCallback, useState } from "react";
import { VideoOff } from "lucide-react";
import type { RobotStateName } from "@/lib/types.ts";

const CAMERAS = ["left", "top", "right"] as const;
const LABELS: Record<string, string> = { left: "Left", top: "Top", right: "Right" };

interface CameraGridProps {
  state: RobotStateName;
  camTs: number | null;
  availableCameras: string[];
}

function CameraPlaceholder({ label }: { label: string }) {
  return (
    <div className="flex aspect-[4/3] flex-col items-center justify-center gap-2 rounded-lg bg-zinc-900">
      <VideoOff className="h-8 w-8 text-zinc-600" />
      <span className="text-xs text-zinc-600">{label}</span>
    </div>
  );
}

function CameraFeed({
  camera,
  camTs,
  available,
}: {
  camera: string;
  camTs: number | null;
  available: boolean;
}) {
  const [errored, setErrored] = useState(false);
  const label = LABELS[camera] ?? camera;

  const handleError = useCallback(() => setErrored(true), []);

  const connected = camTs !== null && available;
  if (!connected || errored) {
    return <CameraPlaceholder label={label} />;
  }

  return (
    <div className="relative overflow-hidden rounded-lg bg-zinc-900">
      <img
        src={`/stream/${camera}?t=${camTs}`}
        alt={`${label} camera`}
        className="aspect-[4/3] w-full object-cover"
        onError={handleError}
      />
      <span className="absolute left-2 top-2 rounded bg-black/60 px-2 py-0.5 text-[11px] font-medium">
        {label}
      </span>
    </div>
  );
}

export function CameraGrid({ state, camTs, availableCameras }: CameraGridProps) {
  const connected = state !== "disconnected";
  return (
    <div className="grid min-h-0 flex-1 grid-cols-3 gap-2 p-3">
      {CAMERAS.map((cam) => (
        <CameraFeed
          key={cam}
          camera={cam}
          camTs={connected ? camTs : null}
          available={availableCameras.includes(cam)}
        />
      ))}
    </div>
  );
}
