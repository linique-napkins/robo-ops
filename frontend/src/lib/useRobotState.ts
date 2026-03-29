import { useCallback, useRef, useSyncExternalStore } from "react";
import * as api from "@/lib/api.ts";
import { DEFAULT_STATE } from "@/lib/types.ts";
import type { ApiResponse, RobotState } from "@/lib/types.ts";

// ---------------------------------------------------------------------------
// Module-level singleton: one WebSocket, one state, many subscribers
// ---------------------------------------------------------------------------

type Listener = () => void;

let currentState: RobotState = { ...DEFAULT_STATE };
let wsConnected = false;
let statusMessage = "Ready";
let inflight = false;

const listeners = new Set<Listener>();

function notify() {
  for (const fn of listeners) fn();
}

function setState(next: RobotState) {
  currentState = next;
  notify();
}

function setWsConnected(v: boolean) {
  wsConnected = v;
  notify();
}

function setStatus(msg: string) {
  statusMessage = msg;
  notify();
}

function setInflight(v: boolean) {
  inflight = v;
  notify();
}

// ---------------------------------------------------------------------------
// WebSocket with exponential backoff reconnection
// ---------------------------------------------------------------------------

let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let backoff = 1000;
const MAX_BACKOFF = 10000;

function connectWs() {
  if (ws) return;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${proto}//${location.host}/ws/state`);

  socket.onopen = () => {
    ws = socket;
    backoff = 1000;
    setWsConnected(true);
    // Full state sync on connect
    api.fetchState().then(setState).catch(() => {});
  };

  socket.onmessage = (e: MessageEvent) => {
    try {
      const data = JSON.parse(e.data as string) as RobotState;
      setState(data);
    } catch {
      // ignore malformed
    }
  };

  socket.onclose = () => {
    ws = null;
    setWsConnected(false);
    scheduleReconnect();
  };

  socket.onerror = () => {
    socket.close();
  };
}

function disconnectWs() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
  setWsConnected(false);
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    backoff = Math.min(backoff * 2, MAX_BACKOFF);
    connectWs();
  }, backoff);
}

// ---------------------------------------------------------------------------
// API action wrapper: updates state from response, shows errors
// ---------------------------------------------------------------------------

async function doAction(
  fn: () => Promise<ApiResponse>,
  label?: string,
): Promise<boolean> {
  if (inflight) return false;
  setInflight(true);
  if (label) setStatus(label);
  try {
    const data = await fn();
    if (data.state) setState(data.state as RobotState);
    if (!data.ok && data.error) setStatus(`Error: ${data.error}`);
    return data.ok;
  } catch (e) {
    setStatus(`Request failed: ${e instanceof Error ? e.message : String(e)}`);
    return false;
  } finally {
    setInflight(false);
  }
}

// ---------------------------------------------------------------------------
// Boot: fetch initial state + connect WS if already connected
// ---------------------------------------------------------------------------

let booted = false;

function boot() {
  if (booted) return;
  booted = true;
  api
    .fetchState()
    .then((s) => {
      setState(s);
      if (s.state !== "disconnected") {
        connectWs();
      }
    })
    .catch(() => {});
}

// ---------------------------------------------------------------------------
// Snapshot getters (for useSyncExternalStore)
// ---------------------------------------------------------------------------

interface Snapshot {
  state: RobotState;
  wsConnected: boolean;
  status: string;
  inflight: boolean;
}

let cachedSnapshot: Snapshot = {
  state: currentState,
  wsConnected,
  status: statusMessage,
  inflight,
};

function getSnapshot(): Snapshot {
  const next: Snapshot = {
    state: currentState,
    wsConnected,
    status: statusMessage,
    inflight,
  };
  // Only return a new reference if something actually changed
  if (
    cachedSnapshot.state !== next.state ||
    cachedSnapshot.wsConnected !== next.wsConnected ||
    cachedSnapshot.status !== next.status ||
    cachedSnapshot.inflight !== next.inflight
  ) {
    cachedSnapshot = next;
  }
  return cachedSnapshot;
}

function subscribe(cb: Listener): () => void {
  listeners.add(cb);
  boot();
  return () => listeners.delete(cb);
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useRobotState() {
  const snapshot = useSyncExternalStore(subscribe, getSnapshot);
  // Stable camera timestamp ref (set once on connect)
  const camTsRef = useRef<number | null>(null);

  const actions = {
    connect: useCallback(async () => {
      const ok = await doAction(() => api.connect(), "Connecting to robot...");
      if (ok) {
        camTsRef.current = Date.now();
        connectWs();
        setStatus("Connected");
      }
    }, []),

    disconnect: useCallback(async () => {
      const ok = await doAction(
        () => api.disconnect(),
        "Disconnecting...",
      );
      if (ok) {
        camTsRef.current = null;
        disconnectWs();
        setStatus("Disconnected");
      }
    }, []),

    stow: useCallback(
      () => doAction(() => api.stow(), "Stowing..."),
      [],
    ),

    togglePause: useCallback(
      () => doAction(() => api.togglePause()),
      [],
    ),

    startReplay: useCallback(
      (name: string, loop: boolean) =>
        doAction(
          () => api.startReplay(name, loop),
          `Starting replay: ${name}`,
        ),
      [],
    ),

    stopReplay: useCallback(
      (name: string) => doAction(() => api.stopReplay(name)),
      [],
    ),

    startInference: useCallback(
      (model: string) =>
        doAction(
          () => api.startInference(model),
          `Loading model: ${model}...`,
        ),
      [],
    ),

    stopInference: useCallback(
      (model: string) => doAction(() => api.stopInference(model)),
      [],
    ),

    startTeleop: useCallback(
      () => doAction(() => api.startTeleop(), "Starting teleop..."),
      [],
    ),

    stopTeleop: useCallback(
      () => doAction(() => api.stopTeleop()),
      [],
    ),

    setStatus: useCallback((msg: string) => setStatus(msg), []),
  };

  return {
    ...snapshot,
    camTs: camTsRef.current,
    api: actions,
  };
}
