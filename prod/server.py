"""
Production demo server for bimanual SO101 robot.

FastAPI application exposing REST endpoints for robot control,
MJPEG camera streaming, and WebSocket connections for state
broadcasts and remote teleoperation.

Usage:
    uv run prod/server.py
"""

import asyncio
import json
import sys
import tomllib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from camera_stream import mjpeg_stream
from inference_backend import LocalInference
from inference_backend import RemoteInference
from operations import InferenceOperation
from operations import ReplayOperation
from operations import TeleopOperation
from robot_manager import RobotManager
from robot_manager import State

# --- Configuration ---

PROD_DIR = Path(__file__).parent
PROD_CONFIG_PATH = PROD_DIR / "config.toml"
RECORDINGS_DIR = PROD_DIR.parent / "demo" / "recordings"
STATIC_DIR = PROD_DIR / "static"


def load_prod_config() -> dict:
    with PROD_CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


prod_config = load_prod_config()


# --- Shared mutable state (avoids module-level globals) ---


class _AppState:
    def __init__(self):
        self.event_loop: asyncio.AbstractEventLoop | None = None
        self.teleop_op: TeleopOperation | None = None
        self.ws_clients: set[WebSocket] = set()


_app = _AppState()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:  # noqa: ARG001
    _app.event_loop = asyncio.get_running_loop()
    yield


# --- App Setup ---

app = FastAPI(title="SO101 Demo Server", lifespan=lifespan)
manager = RobotManager()


async def broadcast_state(state_dict: dict) -> None:
    """Send state update to all connected WebSocket clients."""
    if not _app.ws_clients:
        return
    data = json.dumps(state_dict)
    dead: set[WebSocket] = set()
    for ws in _app.ws_clients.copy():
        try:
            await ws.send_text(data)
        except Exception:
            dead.add(ws)
    _app.ws_clients.difference_update(dead)


def _handle_state_change(state_dict: dict) -> None:
    """Thread-safe callback invoked from the control thread."""
    loop = _app.event_loop
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast_state(state_dict), loop)


manager.on_state_change = _handle_state_change


# --- Static Files / Frontend ---

FRONTEND_DIR = PROD_DIR.parent / "frontend" / "dist"

if FRONTEND_DIR.exists():
    # Serve Vite build assets
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
else:
    # Fallback to vanilla UI
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- REST API ---


@app.get("/api/state")
async def get_state():
    models = {}
    for key, cfg in prod_config.get("models", {}).items():
        models[key] = cfg.get("display_name", key)

    recordings = []
    if RECORDINGS_DIR.exists():
        recordings = sorted(p.stem for p in RECORDINGS_DIR.glob("*.json"))

    state = manager.get_state()
    state["available_models"] = models
    state["available_recordings"] = recordings
    state["available_cameras"] = manager.get_camera_names()
    return state


@app.post("/api/connect")
async def connect():
    try:
        await manager.connect()
        return {"ok": True, "state": manager.get_state()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/disconnect")
async def disconnect():
    try:
        await manager.disconnect()
        return {"ok": True, "state": manager.get_state()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/stow")
async def stow_endpoint():
    try:
        await manager.stow_robot()
        return {"ok": True, "state": manager.get_state()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/pause")
async def pause():
    try:
        paused = await manager.toggle_pause()
        return {"ok": True, "paused": paused, "state": manager.get_state()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/replay/{name}")
async def start_replay(name: str, loop: bool = False):
    recording_path = RECORDINGS_DIR / f"{name}.json"
    if not recording_path.exists():
        return JSONResponse(
            {"ok": False, "error": f"Recording '{name}' not found"},
            status_code=404,
        )
    try:
        op = ReplayOperation(recording_path, loop=loop)
        await manager.start_operation(op, State.REPLAYING)
        return {"ok": True, "state": manager.get_state()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/replay/{name}/stop")
async def stop_replay(name: str):  # noqa: ARG001
    try:
        await manager.stop_operation()
        return {"ok": True, "state": manager.get_state()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/infer/{model}")
async def start_inference(model: str):
    models = prod_config.get("models", {})
    if model not in models:
        return JSONResponse(
            {"ok": False, "error": f"Model '{model}' not found"},
            status_code=404,
        )
    model_cfg = models[model]
    try:
        backend_type = model_cfg.get("backend", "local")
        backend = RemoteInference() if backend_type == "remote" else LocalInference()
        backend.load(model_cfg)
        op = InferenceOperation(backend)
        await manager.start_operation(op, State.INFERRING)
        return {"ok": True, "state": manager.get_state()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/infer/{model}/stop")
async def stop_inference(model: str):  # noqa: ARG001
    try:
        await manager.stop_operation()
        return {"ok": True, "state": manager.get_state()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/teleop/start")
async def start_teleop():
    try:
        _app.teleop_op = TeleopOperation()
        await manager.start_operation(_app.teleop_op, State.TELEOP)
        return {"ok": True, "state": manager.get_state()}
    except Exception as e:
        _app.teleop_op = None
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/teleop/stop")
async def stop_teleop():
    try:
        await manager.stop_operation()
        _app.teleop_op = None
        return {"ok": True, "state": manager.get_state()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# --- Camera Streaming ---


@app.get("/stream/{camera}")
async def camera_stream(camera: str):
    """MJPEG stream for a single camera. Use as <img src="/stream/left">."""
    return StreamingResponse(
        mjpeg_stream(lambda cam=camera: manager.get_latest_frame(cam)),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# --- WebSocket: State Broadcast ---


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    await websocket.accept()
    _app.ws_clients.add(websocket)
    try:
        await websocket.send_text(json.dumps(manager.get_state()))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _app.ws_clients.discard(websocket)


# --- WebSocket: Remote Teleoperation ---


@app.websocket("/ws/teleop")
async def ws_teleop(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if _app.teleop_op:
                msg = json.loads(data)
                _app.teleop_op.update_action(msg.get("positions", msg))
    except WebSocketDisconnect:
        pass


# --- SPA Fallback (must be after all /api, /stream, /ws routes) ---

if FRONTEND_DIR.exists():

    @app.get("/{path:path}")
    async def spa_fallback(path: str):
        file = FRONTEND_DIR / path
        if file.exists() and file.is_file():
            return FileResponse(str(file))
        # No-cache so browser always gets latest index.html after rebuilds
        return FileResponse(
            str(FRONTEND_DIR / "index.html"),
            headers={"Cache-Control": "no-cache"},
        )

else:

    @app.get("/")
    async def root():
        return FileResponse(str(STATIC_DIR / "index.html"))


# --- Entry Point ---

if __name__ == "__main__":
    import asyncio

    from hypercorn.asyncio import serve
    from hypercorn.config import Config as HyperConfig

    server_cfg = prod_config.get("server", {})
    hconfig = HyperConfig()
    hconfig.bind = [f"{server_cfg.get('host', '0.0.0.0')}:{server_cfg.get('port', 8000)}"]

    # TLS + HTTP/2 via Tailscale certs
    cert_dir = PROD_DIR / "certs"
    cert_file = cert_dir / "nvd-compute.crt"
    key_file = cert_dir / "nvd-compute.key"
    if cert_file.exists() and key_file.exists():
        hconfig.certfile = str(cert_file)
        hconfig.keyfile = str(key_file)

    asyncio.run(serve(app, hconfig))
