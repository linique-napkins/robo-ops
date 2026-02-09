"""
Capture a photo from each camera defined in config.toml and save to outputs/.

Takes a full-resolution frame from each camera, applies a center crop to the
configured width x height, and saves both the original and cropped images.

Usage:
    uv run utils/test_cameras.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from lib.config import get_camera_config
from lib.config import load_config

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "camera_test"


def center_crop(frame, target_w: int, target_h: int):
    """Center crop a frame to target_w x target_h."""
    h, w = frame.shape[:2]
    x = (w - target_w) // 2
    y = (h - target_h) // 2
    return frame[y : y + target_h, x : x + target_w]


def main() -> None:
    config = load_config()
    cameras = get_camera_config(config)

    if not cameras:
        print("No cameras configured in config.toml")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving photos to {OUTPUT_DIR}\n")

    for name, cam in cameras.items():
        path = cam["path"]
        target_w = cam["width"]
        target_h = cam["height"]
        fourcc = cam.get("fourcc")

        print(f"[{name}] Opening device {path}...")
        cap = cv2.VideoCapture(path)

        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        if not cap.isOpened():
            print(f"[{name}] FAILED to open camera at {path}")
            continue

        # Read at native resolution
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"[{name}] FAILED to read frame")
            continue

        native_h, native_w = frame.shape[:2]
        print(f"[{name}] Native: {native_w}x{native_h}")

        # Save original
        orig_path = OUTPUT_DIR / f"{name}_original.png"
        cv2.imwrite(str(orig_path), frame)
        print(f"[{name}] Saved original: {orig_path}")

        # Center crop to configured dimensions
        if target_w > native_w or target_h > native_h:
            print(
                f"[{name}] WARNING: crop {target_w}x{target_h} exceeds"
                f" native {native_w}x{native_h}, skipping crop"
            )
            continue

        cropped = center_crop(frame, target_w, target_h)
        crop_path = OUTPUT_DIR / f"{name}_cropped_{target_w}x{target_h}.png"
        cv2.imwrite(str(crop_path), cropped)
        print(f"[{name}] Saved cropped: {crop_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
