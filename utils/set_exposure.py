"""
Re-apply camera exposure settings to all configured cameras.

v4l2 exposure/focus settings are volatile — they reset on USB reconnects,
sleep/wake, or just drift over time. This script reads the camera paths
already in config.toml and re-applies exposure settings without re-doing
the full discovery/assignment flow.

Usage:
    uv run utils/set_exposure.py              # Apply to all cameras
    uv run utils/set_exposure.py --preview    # Apply + show live preview to verify
    uv run utils/set_exposure.py --camera top # Apply to one camera only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import typer

from lib.config import get_camera_config
from lib.config import load_config
from utils.find_cameras import configure_exposure
from utils.find_cameras import v4l2_get_controls

WINDOW_NAME = "Exposure Preview"

app = typer.Typer()


def resolve_device(path: str) -> str:
    """Resolve a /dev/v4l/by-path/ symlink to its /dev/videoN target."""
    p = Path(path)
    if p.is_symlink():
        return str(p.resolve())
    return path


def show_preview(cameras: dict[str, dict]) -> None:
    """Show a live tiled preview of all cameras so you can verify exposure."""
    caps: dict[str, cv2.VideoCapture] = {}
    for name, cam in cameras.items():
        cap = cv2.VideoCapture(cam["path"])
        if not cap.isOpened():
            print(f"  WARNING: could not open {name} ({cam['path']}) for preview")
            continue
        fourcc = cam.get("fourcc")
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.get("width", 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("height", 480))
        cap.set(cv2.CAP_PROP_FPS, cam.get("fps", 30))
        caps[name] = cap

    if not caps:
        print("No cameras available for preview.")
        return

    print("\nShowing preview — press 'q' to close.\n")
    while True:
        frames = []
        for name, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frames.append(frame)

        if frames:
            # Stack frames horizontally
            tiled = cv2.hconcat(frames)
            cv2.imshow(WINDOW_NAME, tiled)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


@app.callback(invoke_without_command=True)
def main(
    camera: str | None = typer.Option(
        None, "--camera", "-c", help="Apply to a single camera role (left, right, top)"
    ),
    preview: bool = typer.Option(False, "--preview", "-p", help="Show live preview after applying"),
) -> None:
    """Re-apply exposure settings to configured cameras."""
    config = load_config()
    all_cameras = get_camera_config(config)

    if camera:
        if camera not in all_cameras:
            print(f"ERROR: camera '{camera}' not found in config.toml")
            print(f"Available cameras: {', '.join(all_cameras)}")
            raise SystemExit(1)
        cameras = {camera: all_cameras[camera]}
    else:
        cameras = all_cameras

    print(f"Applying exposure settings to {len(cameras)} camera(s)...\n")

    for role, cam in cameras.items():
        device = resolve_device(cam["path"])
        controls = v4l2_get_controls(device)
        if not controls:
            print(f"  [{role}] WARNING: no v4l2 controls found at {device}")
            print(f"          (path: {cam['path']})")
            continue
        configure_exposure(device, role)

    print("\nDone.")

    if preview:
        show_preview(cameras)


if __name__ == "__main__":
    app()
