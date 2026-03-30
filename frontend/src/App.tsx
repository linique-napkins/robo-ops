import { CameraGrid } from "@/components/CameraGrid.tsx";
import { ControlBar } from "@/components/ControlBar.tsx";
import { Header } from "@/components/Header.tsx";
import { StatusBar } from "@/components/StatusBar.tsx";
import { useRobotState } from "@/lib/useRobotState.ts";

export default function App() {
  const { state, wsConnected, status, inflight, camTs, api } = useRobotState();

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#f5f0e8] text-zinc-900">
      <Header
        state={state.state}
        paused={state.paused}
        operation={state.operation}
      />
      <CameraGrid
        cameras={state.available_cameras ?? []}
        camTs={state.state !== "disconnected" ? camTs : null}
      />
      <ControlBar state={state} inflight={inflight} api={api} />
      <StatusBar message={status} wsConnected={wsConnected} />
    </div>
  );
}
