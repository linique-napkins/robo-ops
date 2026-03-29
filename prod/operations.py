"""
Operation implementations for the robot control loop.

Each operation provides a step() method that returns an action dict (or None)
given the current robot observation. The robot manager calls these at 30fps.
"""

import json
import threading
import time
from pathlib import Path
from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class Operation(Protocol):
    """Protocol for robot control operations."""

    name: str
    paused: bool

    def setup(self, robot) -> None: ...
    def step(self, observation: dict) -> dict | None: ...
    def teardown(self) -> None: ...


class ReplayOperation:
    """Replays a recorded demo JSON file frame by frame."""

    name = "replay"

    def __init__(self, recording_path: Path, loop: bool = False):
        recording = json.loads(recording_path.read_text())
        self.frames: list[dict] = recording["frames"]
        self.loop = loop
        self.paused = False
        self.finished = False
        self._index = 0

    def setup(self, robot) -> None:  # noqa: ARG002
        self._index = 0
        self.finished = False

    def step(self, observation: dict) -> dict | None:  # noqa: ARG002
        if self._index >= len(self.frames):
            if self.loop:
                self._index = 0
            else:
                self.finished = True
                return None
        frame = self.frames[self._index]
        self._index += 1
        return frame

    def teardown(self) -> None:
        pass

    @property
    def progress(self) -> float:
        if not self.frames:
            return 0.0
        return self._index / len(self.frames)

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    @property
    def current_frame(self) -> int:
        return self._index


class InferenceOperation:
    """Runs a loaded inference backend to generate actions from observations."""

    name = "inference"

    def __init__(self, backend):
        self._backend = backend
        self._robot_name: str = ""
        self.paused = False
        self.finished = False

    def setup(self, robot) -> None:
        self._robot_name = robot.name

    def step(self, observation: dict) -> dict | None:
        return self._backend.predict(observation, self._robot_name)

    def teardown(self) -> None:
        pass


class TeleopOperation:
    """Receives remote leader arm positions and forwards them as actions."""

    name = "teleop"

    def __init__(self, timeout: float = 2.0):
        self._latest_action: dict | None = None
        self._lock = threading.Lock()
        self._last_receive_time: float = 0
        self._timeout = timeout
        self.paused = False
        self.finished = False

    def setup(self, robot) -> None:  # noqa: ARG002
        with self._lock:
            self._latest_action = None
            self._last_receive_time = time.time()

    def update_action(self, action: dict) -> None:
        """Called from WebSocket handler with latest leader positions."""
        with self._lock:
            self._latest_action = action
            self._last_receive_time = time.time()

    def step(self, observation: dict) -> dict | None:  # noqa: ARG002
        with self._lock:
            if self._latest_action is None:
                return None
            if time.time() - self._last_receive_time > self._timeout:
                return None
            return dict(self._latest_action)

    def teardown(self) -> None:
        with self._lock:
            self._latest_action = None

    @property
    def connected(self) -> bool:
        """Whether a teleop client is actively sending commands."""
        with self._lock:
            if self._latest_action is None:
                return False
            return time.time() - self._last_receive_time <= self._timeout
