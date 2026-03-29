import type { ApiResponse, RobotState } from "@/lib/types.ts";

async function request(method: string, path: string): Promise<ApiResponse> {
  const res = await fetch(`/api/${path}`, { method });
  return (await res.json()) as ApiResponse;
}

export async function fetchState(): Promise<RobotState> {
  const res = await fetch("/api/state");
  return (await res.json()) as RobotState;
}

export async function connect(): Promise<ApiResponse> {
  return request("POST", "connect");
}

export async function disconnect(): Promise<ApiResponse> {
  return request("POST", "disconnect");
}

export async function stow(): Promise<ApiResponse> {
  return request("POST", "stow");
}

export async function togglePause(): Promise<ApiResponse> {
  return request("POST", "pause");
}

export async function startReplay(
  name: string,
  loop: boolean,
): Promise<ApiResponse> {
  const qs = loop ? "?loop=true" : "";
  return request("POST", `replay/${encodeURIComponent(name)}${qs}`);
}

export async function stopReplay(name: string): Promise<ApiResponse> {
  return request(
    "POST",
    `replay/${encodeURIComponent(name || "_")}/stop`,
  );
}

export async function startInference(model: string): Promise<ApiResponse> {
  return request("POST", `infer/${encodeURIComponent(model)}`);
}

export async function stopInference(model: string): Promise<ApiResponse> {
  return request(
    "POST",
    `infer/${encodeURIComponent(model || "_")}/stop`,
  );
}

export async function startTeleop(): Promise<ApiResponse> {
  return request("POST", "teleop/start");
}

export async function stopTeleop(): Promise<ApiResponse> {
  return request("POST", "teleop/stop");
}
