"""
Robot lifecycle management, state machine, and control thread coordination.

Owns the robot object, camera configs, state machine, the synchronous 30fps
control thread, and the shared JPEG frame buffer for camera streaming.
"""

import threading
import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from lerobot.utils.robot_utils import precise_sleep

from lib.config import get_camera_config
from lib.config import load_config
from lib.robots import build_camera_configs
from lib.robots import get_bimanual_follower
from lib.stow import stow
from lib.stow import stow_and_disconnect
from utils.find_cameras import configure_exposure


class State(Enum):
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    REPLAYING = "replaying"
    INFERRING = "inferring"
    TELEOP = "teleop"
    STOWING = "stowing"


class RobotManager:
    """Manages robot lifecycle, state transitions, and the control thread."""

    def __init__(self):
        self.state = State.DISCONNECTED
        self.robot = None
        self._operation = None
        self._config: dict | None = None
        self._cameras_cfg: dict | None = None

        # Frame buffer: camera_name ("left", "right", "top") -> JPEG bytes
        self._frame_buffer: dict[str, bytes] = {}
        self._frame_lock = threading.Lock()

        # Control thread
        self._control_thread: threading.Thread | None = None
        self._running = False

        # Callback for state change broadcast (set by server.py)
        self.on_state_change = None

    def _set_state(self, new_state: State) -> None:
        self.state = new_state
        if self.on_state_change:
            self.on_state_change(self.get_state())

    def get_state(self) -> dict:
        return {
            "state": self.state.value,
            "paused": self._operation.paused if self._operation else False,
            "operation": self._operation.name if self._operation else None,
        }

    def get_latest_frame(self, camera: str) -> bytes | None:
        """Get the latest JPEG frame for a camera, or None if unavailable."""
        with self._frame_lock:
            return self._frame_buffer.get(camera)

    def get_camera_names(self) -> list[str]:
        """Return configured camera names (e.g. ['left', 'right', 'top'])."""
        return list(self._cameras_cfg.keys()) if self._cameras_cfg else []

    async def connect(self) -> None:
        """Connect to robot hardware and start the control loop."""
        if self.state != State.DISCONNECTED:
            raise RuntimeError(f"Cannot connect in state {self.state.value}")

        self._config = load_config()

        # Validate follower ports (leader is on a separate machine)
        for arm in ["left", "right"]:
            port = self._config.get("follower", {}).get(arm, {}).get("port")
            if not port:
                raise ValueError(f"No port configured for {arm} follower in config.toml")

        self._cameras_cfg = get_camera_config(self._config)

        # Apply v4l2 exposure settings for OpenCV cameras (skip RealSense)
        for name, cam in self._cameras_cfg.items():
            if cam["type"] == "realsense":
                continue
            device = Path(cam["path"]).resolve()
            configure_exposure(str(device), name)

        camera_configs = build_camera_configs(self._cameras_cfg)

        self.robot = get_bimanual_follower(self._config, cameras=camera_configs)
        try:
            self.robot.connect(calibrate=False)
        except Exception:
            # Clean up partially-opened cameras/motors so retry works
            try:
                self.robot.disconnect()
            except Exception:
                pass
            self.robot = None
            raise

        self._start_control_thread()
        self._set_state(State.IDLE)

    async def disconnect(self) -> None:
        """Stop operations, stow, and disconnect robot."""
        if self.state == State.DISCONNECTED:
            return

        if self._operation:
            await self.stop_operation()

        self._stop_control_thread()

        if self.robot and self.robot.is_connected:
            stow_and_disconnect(self.robot)

        self.robot = None
        self._frame_buffer.clear()
        self._set_state(State.DISCONNECTED)

    async def start_operation(self, operation, target_state: State) -> None:
        """Start a new operation (replay, inference, or teleop)."""
        if self.state != State.IDLE:
            raise RuntimeError(f"Cannot start operation in state {self.state.value}")

        self._operation = operation
        self._operation.setup(self.robot)
        self._set_state(target_state)

    async def stop_operation(self) -> None:
        """Stop the current operation and return to IDLE."""
        if not self._operation:
            return
        self._operation.teardown()
        self._operation = None
        if self.state != State.DISCONNECTED:
            self._set_state(State.IDLE)

    async def stow_robot(self) -> None:
        """Stow arms to safe position."""
        if self.state == State.DISCONNECTED:
            raise RuntimeError("Robot not connected")

        if self._operation:
            await self.stop_operation()

        self._set_state(State.STOWING)

        # Pause control loop so stow() can use the USB bus exclusively
        self._stop_control_thread()
        stow(self.robot)
        self._start_control_thread()

        self._set_state(State.IDLE)

    async def toggle_pause(self) -> bool:
        """Toggle pause on the active operation. Returns new paused state."""
        if not self._operation:
            return False
        self._operation.paused = not self._operation.paused
        if self.on_state_change:
            self.on_state_change(self.get_state())
        return self._operation.paused

    def _start_control_thread(self) -> None:
        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

    def _stop_control_thread(self) -> None:
        self._running = False
        if self._control_thread:
            self._control_thread.join(timeout=3.0)
            self._control_thread = None

    def _control_loop(self) -> None:
        """Synchronous 30fps control loop running in a dedicated thread.

        Thread 2 in the architecture:
        - robot.get_observation() -> joints + camera numpy arrays
        - operation.step() -> action (varies by mode)
        - robot.send_action(action)
        - JPEG-encode camera frames -> write to shared frame buffer
        """
        fps = 30
        while self._running:
            loop_start = time.perf_counter()
            try:
                obs = self.robot.get_observation()
                self._update_frame_buffer(obs)

                if self._operation and not self._operation.paused:
                    action = self._operation.step(obs)
                    if action:
                        self.robot.send_action(action)

                    if getattr(self._operation, "finished", False):
                        self._operation.teardown()
                        self._operation = None
                        self._set_state(State.IDLE)

            except Exception as e:
                print(f"Control loop error: {e}")

            dt = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt, 0.0))

    # Arm cameras are mounted sideways — rotate 90° CW for human-aligned view
    _ROTATE_CAMERAS = {"left", "right"}

    def _update_frame_buffer(self, obs: dict) -> None:
        """Extract camera frames from observation and JPEG encode to buffer.

        Observation keys may be prefixed with arm name (e.g. 'left_top_cam'
        instead of 'top_cam'), so we match by suffix.
        """
        for obs_key, value in obs.items():
            if not isinstance(value, np.ndarray) or value.ndim != 3:
                continue
            for camera_name in self._cameras_cfg:
                if obs_key.endswith(f"{camera_name}_cam"):
                    bgr = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
                    if camera_name in self._ROTATE_CAMERAS:
                        bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif camera_name == "top":
                        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
                    _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    with self._frame_lock:
                        self._frame_buffer[camera_name] = jpeg.tobytes()
                    break
