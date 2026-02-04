"""
Test script for URDF visualization in Rerun.

Animates the robot arms through a range of motion to verify the kinematic chain works correctly.

Usage:
    uv run utils/test_urdf_viz.py
"""

import math
import sys
import time
from pathlib import Path

import rerun as rr

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import get_urdf_config
from lib.config import load_config
from lib.urdf_viz import init_rerun_with_urdf


def main() -> None:
    """Run URDF visualization test with animated joints."""
    print("=== URDF Visualization Test ===\n")

    # Load config
    try:
        config = load_config()
        urdf_cfg = get_urdf_config(config)
    except FileNotFoundError:
        print("Warning: config.toml not found, using defaults")
        urdf_cfg = {
            "path": None,
            "left_offset": (-0.2, 0.0, 0.0),
            "right_offset": (0.2, 0.0, 0.0),
            "left_rotation": 0.0,
            "right_rotation": 0.0,
        }

    # Initialize Rerun with URDF
    print("Initializing Rerun with URDF visualization...")
    visualizer = init_rerun_with_urdf(
        session_name="urdf_test",
        urdf_path=urdf_cfg["path"],
        left_offset=urdf_cfg["left_offset"],
        right_offset=urdf_cfg["right_offset"],
        left_rotation_deg=urdf_cfg["left_rotation"],
        right_rotation_deg=urdf_cfg["right_rotation"],
    )

    if not visualizer:
        print("ERROR: Failed to initialize URDF visualization!")
        print("Make sure the SO-ARM100 submodule is initialized:")
        print("  git submodule update --init --recursive")
        return

    print("URDF visualization initialized!")
    print(f"Link paths (left arm): {visualizer._link_paths['left']}")
    print(f"Link paths (right arm): {visualizer._link_paths['right']}")
    print("\nAnimating robot arms... Press Ctrl+C to exit\n")

    # Joint names that match the URDF
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    # Animation parameters (degrees)
    joint_ranges = {
        "shoulder_pan": (-45, 45),
        "shoulder_lift": (-30, 60),
        "elbow_flex": (-90, 0),
        "wrist_flex": (-45, 45),
        "wrist_roll": (-90, 90),
        "gripper": (0, 50),
    }

    try:
        t = 0.0
        dt = 0.05  # 20 Hz
        while True:
            # Build observation dict with animated joint values
            observation = {}

            for joint in joints:
                low, high = joint_ranges[joint]
                mid = (low + high) / 2
                amp = (high - low) / 2

                # Different phase for each joint to make it look interesting
                phase = joints.index(joint) * 0.5
                value = mid + amp * math.sin(t + phase)

                # Set for both arms (mirrored for some joints)
                # Use .pos suffix to match real robot observation format
                observation[f"left_{joint}.pos"] = value
                # Mirror shoulder_pan and wrist_roll for right arm
                if joint in ("shoulder_pan", "wrist_roll"):
                    observation[f"right_{joint}.pos"] = -value
                else:
                    observation[f"right_{joint}.pos"] = value

            # Log to Rerun
            rr.set_time_seconds("time", t)
            visualizer.log_robot_state(observation, use_degrees=True)

            # Also log joint values as scalars for debugging
            for key, value in observation.items():
                if key.startswith("left_"):
                    joint_name = key.removeprefix("left_").removesuffix(".pos")
                    rr.log(f"joints/left/{joint_name}", rr.Scalars(value))
                elif key.startswith("right_"):
                    joint_name = key.removeprefix("right_").removesuffix(".pos")
                    rr.log(f"joints/right/{joint_name}", rr.Scalars(value))

            t += dt
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nTest stopped.")


if __name__ == "__main__":
    main()
