"""
Discover cameras via stable /dev/v4l/ paths and assign them to config roles.

Opens each camera one at a time, shows a live preview so you can identify it,
then asks you to assign it (left / right / top / skip). Updates config.toml
with stable device paths that won't change across reboots, and configures
camera exposure/focus settings via v4l2-ctl.

Uses /dev/v4l/by-path/ (physical USB port) as the primary source since cameras
with identical serial numbers (like the Innomaker U20CAMs) only get a single
entry in /dev/v4l/by-id/. by-path is stable as long as cameras stay in the
same physical USB ports.

Usage:
    uv run utils/find_cameras.py           # Interactive: preview + assign
    uv run utils/find_cameras.py --list    # Just list detected cameras
"""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import typer

from lib.config import CONFIG_PATH

V4L_BY_PATH = Path("/dev/v4l/by-path")
V4L_BY_ID = Path("/dev/v4l/by-id")
CAMERA_ROLES = ["left", "right", "top"]
WINDOW_NAME = "Camera Preview"


def get_stable_cameras() -> list[dict]:
    """Find all cameras with stable /dev/v4l/ paths.

    Prefers /dev/v4l/by-path/ (physical USB port) over /dev/v4l/by-id/ because
    cameras with identical serial numbers only get one by-id entry. by-path
    gives a unique entry per physical port.

    For each by-path entry, also looks up the by-id name (if one exists) to
    show a human-friendly device name.

    Returns list of dicts with 'path' (stable symlink), 'target' (resolved
    /dev/videoN), and 'name' (human-readable label).
    """
    if not V4L_BY_PATH.exists():
        print("ERROR: /dev/v4l/by-path/ does not exist.")
        print("Make sure USB cameras are plugged in and v4l2 is available.")
        raise SystemExit(1)

    # Build a map from resolved device (e.g. /dev/video0) -> by-id name
    by_id_names: dict[str, str] = {}
    if V4L_BY_ID.exists():
        for entry in V4L_BY_ID.iterdir():
            if "video-index0" in entry.name or entry.name.endswith("-index0"):
                by_id_names[str(entry.resolve())] = entry.name

    cameras = []
    for entry in sorted(V4L_BY_PATH.iterdir()):
        # Only keep primary capture endpoints (video-index0), skip metadata nodes
        if not entry.name.endswith("-video-index0"):
            continue
        # Skip the usbv2/usbv3 duplicates — prefer the shorter "usb-" variant
        if "-usbv2-" in entry.name or "-usbv3-" in entry.name:
            continue

        target = str(entry.resolve())
        # Use by-id name if available for a friendlier display, fall back to by-path name
        display_name = by_id_names.get(target, entry.name)

        cameras.append(
            {
                "path": str(entry),
                "target": target,
                "name": display_name,
            }
        )

    return cameras


def v4l2_get_controls(device: str) -> set[str]:
    """Query available v4l2 control names for a device."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--list-ctrls"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    controls = set()
    for raw_line in result.stdout.splitlines():
        stripped = raw_line.strip()
        is_header = stripped.startswith("User Controls") or stripped.startswith("Camera Controls")
        if not stripped or is_header:
            continue
        parts = stripped.split()
        if parts:
            controls.add(parts[0])
    return controls


def v4l2_set(device: str, ctrl: str, value: int | str) -> bool:
    """Set a v4l2 control on a device. Returns True on success."""
    try:
        subprocess.run(
            ["v4l2-ctl", "-d", device, "--set-ctrl", f"{ctrl}={value}"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def has_focus_controls(device: str) -> bool:
    """Check if a camera has focus controls (used to detect BRIO-type cameras)."""
    return bool(v4l2_get_controls(device) & {"focus_absolute", "focus_automatic_continuous"})


def configure_exposure(device: str, role: str) -> None:
    """Configure exposure and focus for cameras that need it.

    Only applies manual exposure to cameras with focus controls (like the
    Logitech BRIO). Cameras without focus controls (like the Innomaker U20CAMs)
    work fine on auto exposure and are left untouched.
    """
    controls = v4l2_get_controls(device)
    if not controls:
        print(f"    WARNING: could not query v4l2 controls for {device}")
        return

    if not has_focus_controls(device):
        print(f"  [{role}] Skipping (auto exposure is fine for this camera)")
        return

    print(f"  [{role}] Configuring exposure on {device}...")

    # Manual exposure (1 = Manual Mode, disables auto)
    if "auto_exposure" in controls:
        v4l2_set(device, "auto_exposure", 1)
        print("    auto_exposure = 1 (manual)")

    # Disable backlight compensation
    if "backlight_compensation" in controls:
        v4l2_set(device, "backlight_compensation", 0)
        print("    backlight_compensation = 0")

    # Disable dynamic framerate (can cause exposure to fluctuate)
    if "exposure_dynamic_framerate" in controls:
        v4l2_set(device, "exposure_dynamic_framerate", 0)
        print("    exposure_dynamic_framerate = 0")

    # Set exposure low enough to avoid blowout under studio lighting
    if "exposure_time_absolute" in controls:
        v4l2_set(device, "exposure_time_absolute", 30)
        print("    exposure_time_absolute = 30")

    # Disable autofocus and set to infinity
    if "focus_automatic_continuous" in controls:
        v4l2_set(device, "focus_automatic_continuous", 0)
        print("    focus_automatic_continuous = 0")
    if "focus_absolute" in controls:
        v4l2_set(device, "focus_absolute", 0)
        print("    focus_absolute = 0 (infinity)")


def show_and_assign(cam: dict, remaining_roles: list[str]) -> str | None:
    """Open a single camera, show live preview, and ask the user to assign a role.

    Only one camera is open at a time so there's no USB bandwidth contention.
    Returns the chosen role name, or None if skipped.
    """
    cap = cv2.VideoCapture(cam["path"])
    if not cap.isOpened():
        print(f"  FAILED to open: {cam['name']}")
        return None

    # Set MJPG compressed format in case this isn't the only device on the bus
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Build the prompt string shown on-screen
    options = "  ".join(f"[{r[0]}] {r}" for r in remaining_roles)
    prompt_text = f"{options}  [s] skip"

    print(f"\n  Previewing: {cam['name']}")
    print(f"  Press a key in the preview window: {prompt_text}")

    # Valid key mappings: first letter of each remaining role, plus 's' for skip
    key_map: dict[int, str | None] = {ord("s"): None}
    for role in remaining_roles:
        key_map[ord(role[0])] = role

    chosen = None
    asked = False
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Overlay device info and key hints
        cv2.putText(frame, cam["name"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, prompt_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in key_map:
            chosen = key_map[key]
            break
        elif key != 255 and not asked:
            # Invalid key pressed — remind once
            valid = ", ".join(chr(k) for k in key_map)
            print(f"    (press one of: {valid})")
            asked = True

    cap.release()
    cv2.destroyAllWindows()

    if chosen:
        print(f"    -> assigned as: {chosen}")
    else:
        print("    -> skipped")

    return chosen


def update_config(assignments: dict[str, str]) -> None:
    """Update camera paths in config.toml, preserving all other content."""
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found.")
        raise SystemExit(1)

    text = CONFIG_PATH.read_text()

    for role, stable_path in assignments.items():
        # Match the path line inside [camera.<role>] section
        # Handles both integer paths (path = 6) and quoted string paths (path = "/dev/...")
        pattern = rf"(\[camera\.{re.escape(role)}\][^\[]*?)path\s*=\s*(?:\d+|\"[^\"]*\")"
        replacement = rf'\1path = "{stable_path}"'
        new_text = re.sub(pattern, replacement, text, flags=re.DOTALL)
        if new_text == text:
            print(f"  WARNING: Could not find [camera.{role}] path entry to update.")
        else:
            text = new_text

    CONFIG_PATH.write_text(text)
    print(f"\nUpdated {CONFIG_PATH}")


app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(
    list_only: bool = typer.Option(False, "--list", "-l", help="Just list cameras, don't assign"),
) -> None:
    """Discover cameras and assign stable paths to config.toml."""
    cameras = get_stable_cameras()

    if not cameras:
        print("No cameras found in /dev/v4l/by-path/")
        print("Make sure USB cameras are plugged in.")
        raise SystemExit(1)

    print(f"\nFound {len(cameras)} camera(s):\n")
    for i, cam in enumerate(cameras):
        print(f"  [{i}] {cam['name']}")
        print(f"       {cam['path']}")
        print(f"       -> {cam['target']}")

    if list_only:
        return

    # Walk through each camera one at a time
    remaining_roles = list(CAMERA_ROLES)
    assignments: dict[str, str] = {}
    # Track role -> resolved /dev/videoN for exposure config
    assigned_devices: dict[str, str] = {}

    print(f"\nAssigning {len(cameras)} camera(s) to roles: {', '.join(CAMERA_ROLES)}")
    print("Each camera will preview one at a time. Press the key shown to assign.\n")

    for cam in cameras:
        if not remaining_roles:
            print("  All roles assigned, skipping remaining cameras.")
            break

        role = show_and_assign(cam, remaining_roles)
        if role:
            assignments[role] = cam["path"]
            assigned_devices[role] = cam["target"]
            remaining_roles.remove(role)

    if not assignments:
        print("\nNo assignments made. Config unchanged.")
        return

    # Summary and confirm
    print("\nWill update config.toml with:")
    for role, path in assignments.items():
        print(f'  [camera.{role}] path = "{path}"')

    confirm = input("\nProceed? [Y/n]: ").strip().lower()
    if confirm in ("", "y", "yes"):
        update_config(assignments)
    else:
        print("Cancelled.")
        return

    # Configure exposure on all assigned cameras
    print("\nConfiguring exposure settings...")
    for role, device in assigned_devices.items():
        configure_exposure(device, role)
    print("\nDone.")


if __name__ == "__main__":
    app()
