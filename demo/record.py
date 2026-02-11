"""
Record a demo movement via teleoperation.

Runs normal teleop (leader drives follower) while saving all actions to a JSON file
for later replay. No cameras, no datasets — just joint positions.

Usage:
    uv run demo/record.py                     # Record to demo/recordings/demo.json
    uv run demo/record.py -o my_movement      # Record to demo/recordings/my_movement.json
    uv run demo/record.py --fps 60            # Record at 60 FPS
"""

import json
import sys
import time
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.utils.robot_utils import precise_sleep

from lib.config import load_config
from lib.config import validate_config
from lib.robots import get_bimanual_follower
from lib.robots import get_bimanual_leader

app = typer.Typer()

RECORDINGS_DIR = Path(__file__).parent / "recordings"


@app.command()
def main(
    output: str = typer.Option(
        "demo", "-o", "--output", help="Recording name (saved to demo/recordings/)"
    ),
    fps: int = typer.Option(30, "--fps", help="Recording FPS"),
) -> None:
    """Record a demo movement via teleoperation."""
    config = load_config()
    validate_config(config)

    robot = get_bimanual_follower(config)
    teleop = get_bimanual_leader(config)

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RECORDINGS_DIR / f"{output}.json"

    frames: list[dict[str, float]] = []

    try:
        print("Connecting...")
        teleop.connect(calibrate=False)
        robot.connect(calibrate=False)
        print(f"Recording at {fps} FPS → {out_path}")
        print("Move the leader arms. Press Ctrl+C to stop and save.\n")

        while True:
            loop_start = time.perf_counter()

            action = teleop.get_action()
            robot.send_action(action)

            frames.append({k: float(v) for k, v in action.items()})

            elapsed = len(frames) / fps
            print(f"\r  Frames: {len(frames)}  Time: {elapsed:.1f}s", end="", flush=True)

            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0.0))

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()

    if frames:
        recording = {"fps": fps, "frames": frames}
        out_path.write_text(json.dumps(recording))
        print(f"Saved {len(frames)} frames to {out_path}")
    else:
        print("No frames recorded.")


if __name__ == "__main__":
    app()
