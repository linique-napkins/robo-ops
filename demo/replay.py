"""
Replay a recorded demo movement on the follower arms.

Reads a JSON recording produced by demo/record.py and sends the actions
to the follower arms at the original FPS.

Usage:
    uv run demo/replay.py                        # Replay demo/recordings/demo.json
    uv run demo/replay.py -i my_movement         # Replay demo/recordings/my_movement.json
    uv run demo/replay.py --loop                  # Loop forever
    uv run demo/replay.py --speed 0.5             # Half speed
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

app = typer.Typer()

RECORDINGS_DIR = Path(__file__).parent / "recordings"


@app.command()
def main(
    input_name: str = typer.Option("demo", "-i", "--input", help="Recording name to replay"),
    speed: float = typer.Option(1.0, "--speed", "-s", help="Playback speed multiplier"),
    loop: bool = typer.Option(False, "--loop/--no-loop", help="Loop playback"),
) -> None:
    """Replay a recorded demo movement on the follower arms."""
    in_path = RECORDINGS_DIR / f"{input_name}.json"
    if not in_path.exists():
        print(f"Recording not found: {in_path}")
        raise typer.Exit(1)

    recording = json.loads(in_path.read_text())
    fps = recording["fps"]
    frames = recording["frames"]
    effective_fps = fps * speed

    print(f"Loaded {len(frames)} frames at {fps} FPS from {in_path}")
    print(f"Playback speed: {speed}x ({effective_fps:.0f} effective FPS)")
    duration = len(frames) / fps / speed
    print(f"Duration: {duration:.1f}s{' (looping)' if loop else ''}")

    config = load_config()
    validate_config(config)
    robot = get_bimanual_follower(config)

    try:
        print("\nConnecting...")
        robot.connect(calibrate=False)
        print("Replaying. Press Ctrl+C to stop.\n")

        while True:
            for i, frame in enumerate(frames):
                loop_start = time.perf_counter()

                robot.send_action(frame)

                elapsed = (i + 1) / fps / speed
                msg = f"\r  Frame {i + 1}/{len(frames)}  Time: {elapsed:.1f}/{duration:.1f}s"
                print(msg, end="", flush=True)

                dt_s = time.perf_counter() - loop_start
                precise_sleep(max(1 / effective_fps - dt_s, 0.0))

            if not loop:
                print("\n\nDone!")
                break

            print("\n  Looping...")

    except KeyboardInterrupt:
        print("\n\nStopped.")

    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    app()
