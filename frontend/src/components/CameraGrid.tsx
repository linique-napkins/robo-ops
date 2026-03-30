import { useEffect, useRef, useState } from "react";
import { VideoOff } from "lucide-react";

interface CameraGridProps {
  cameras: string[];
  camTs: number | null;
}

function CameraPlaceholder({ label }: { label: string }) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-2 rounded-lg bg-zinc-200">
      <VideoOff className="h-8 w-8 text-zinc-400" />
      <span className="text-xs text-zinc-400">{label}</span>
    </div>
  );
}

function CameraFeed({ camera, camTs }: { camera: string; camTs: number | null }) {
  const [streamTs, setStreamTs] = useState<number | null>(camTs);
  const retryTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync with parent camTs changes (connect/disconnect)
  useEffect(() => {
    setStreamTs(camTs);
    if (retryTimer.current) {
      clearTimeout(retryTimer.current);
      retryTimer.current = null;
    }
  }, [camTs]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (retryTimer.current) clearTimeout(retryTimer.current);
    };
  }, []);

  const handleError = () => {
    // Retry after 2s with a new timestamp to force a fresh connection
    if (camTs !== null && !retryTimer.current) {
      retryTimer.current = setTimeout(() => {
        retryTimer.current = null;
        setStreamTs(Date.now());
      }, 2000);
    }
  };

  if (streamTs === null) {
    return <CameraPlaceholder label={camera} />;
  }

  return (
    <div className="relative h-full w-full overflow-hidden rounded-lg bg-zinc-200">
      <img
        src={`/stream/${camera}?t=${streamTs}`}
        alt={`${camera} camera`}
        className="h-full w-full object-contain"
        onError={handleError}
      />
      <span className="absolute left-2 top-2 rounded bg-black/60 px-2 py-0.5 text-[11px] font-medium text-white">
        {camera}
      </span>
    </div>
  );
}

export function CameraGrid({ cameras, camTs }: CameraGridProps) {
  const portrait = cameras.filter((c) => c === "left" || c === "right");
  const landscape = cameras.filter((c) => c !== "left" && c !== "right");

  const leftArm = portrait.find((c) => c === "left");
  const rightArm = portrait.find((c) => c === "right");

  return (
    <div className="flex min-h-0 flex-1 gap-1.5 overflow-hidden px-2 py-1.5">
      {leftArm && (
        <div className="flex w-[28%] shrink-0">
          <CameraFeed camera={leftArm} camTs={camTs} />
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col gap-1">
        {landscape.map((cam) => (
          <div key={cam} className="min-h-0 flex-1">
            <CameraFeed camera={cam} camTs={camTs} />
          </div>
        ))}
      </div>

      {rightArm && (
        <div className="flex w-[28%] shrink-0">
          <CameraFeed camera={rightArm} camTs={camTs} />
        </div>
      )}
    </div>
  );
}
