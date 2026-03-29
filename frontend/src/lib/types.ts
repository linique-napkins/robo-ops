export type RobotStateName =
  | "disconnected"
  | "idle"
  | "replaying"
  | "inferring"
  | "teleop"
  | "stowing";

export interface RobotState {
  state: RobotStateName;
  paused: boolean;
  operation: string | null;
  available_models: Record<string, string>;
  available_recordings: string[];
  available_cameras: string[];
}

export interface ApiResponse {
  ok: boolean;
  error?: string;
  state?: RobotState;
  paused?: boolean;
}

export const DEFAULT_STATE: RobotState = {
  state: "disconnected",
  paused: false,
  operation: null,
  available_models: {},
  available_recordings: [],
  available_cameras: [],
};
