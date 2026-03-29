import { ConnectionPanel } from "@/components/ConnectionPanel.tsx";
import { InferencePanel } from "@/components/InferencePanel.tsx";
import { ReplayPanel } from "@/components/ReplayPanel.tsx";
import { TeleopPanel } from "@/components/TeleopPanel.tsx";
import type { RobotState } from "@/lib/types.ts";
import type { useRobotState } from "@/lib/useRobotState.ts";

type Api = ReturnType<typeof useRobotState>["api"];

interface ControlBarProps {
  state: RobotState;
  inflight: boolean;
  api: Api;
}

export function ControlBar({ state, inflight, api }: ControlBarProps) {
  return (
    <div className="flex shrink-0 gap-2 px-3 pb-2">
      <ConnectionPanel
        state={state.state}
        paused={state.paused}
        inflight={inflight}
        onConnect={api.connect}
        onDisconnect={api.disconnect}
        onStow={api.stow}
        onTogglePause={api.togglePause}
      />
      <ReplayPanel
        state={state.state}
        inflight={inflight}
        recordings={state.available_recordings}
        onStartReplay={api.startReplay}
        onStopReplay={api.stopReplay}
      />
      <InferencePanel
        state={state.state}
        inflight={inflight}
        models={state.available_models}
        onStartInference={api.startInference}
        onStopInference={api.stopInference}
      />
      <TeleopPanel
        state={state.state}
        inflight={inflight}
        onStartTeleop={api.startTeleop}
        onStopTeleop={api.stopTeleop}
      />
    </div>
  );
}
