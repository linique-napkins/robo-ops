"""
Capture a photo from each camera index and each configured camera.

First scans indices 0-9, saving a photo from every camera that responds.
Then tests each camera defined in config.toml with its configured settings.

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
MAX_INDEX = 10


def center_crop(frame, target_w: int, target_h: int):
    """Center crop a frame to target_w x target_h."""
    h, w = frame.shape[:2]
    x = (w - target_w) // 2
    y = (h - target_h) // 2
    return frame[y : y + target_h, x : x + target_w]


def scan_all_indices() -> None:
    """Try every index 0..MAX_INDEX, save a photo from each that works."""
    print(f"=== Scanning camera indices 0-{MAX_INDEX - 1} ===\n")
    found = 0
    for idx in range(MAX_INDEX):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"  index {idx}: opened but failed to read frame")
            continue
        h, w = frame.shape[:2]
        out_path = OUTPUT_DIR / f"index_{idx}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"  index {idx}: {w}x{h} -> {out_path}")
        found += 1
    if found == 0:
        print("  No cameras found at any index!")
    print()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving photos to {OUTPUT_DIR}\n")

    # Always scan all indices first
    scan_all_indices()

    # Then test configured cameras
    config = load_config()
    cameras = get_camera_config(config)

    if not cameras:
        print("No cameras configured in config.toml")
        sys.exit(0)

    print("=== Testing configured cameras ===\n")

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
