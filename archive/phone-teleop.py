"""
Phone Teleop Debug Tool
=======================

Control robot arm joints using your phone's orientation.
Useful for testing and debugging individual motors.

Setup
-----
iOS:
    1. Install "HEBI Mobile I/O" app from App Store
    2. Open the app on your phone
    3. Make sure phone and computer are on same WiFi network
    4. Run this script

Configuration
-------------
Edit the variables at the top of this file:
    ARM = "right"     # or "left"

Controls
--------
    - Hold B1 to enable control
    - Tilt phone forward/back: shoulder_lift + elbow_flex
    - Tilt phone left/right: shoulder_pan
    - Rotate phone (roll): wrist_roll
    - Twist phone (yaw): wrist_flex
    - A3 slider: gripper

Usage
-----
    uv run utils/phone-teleop.py

Press Ctrl+C to exit.
"""

import time
import tomllib
from pathlib import Path
from typing import Any

import hebi
import numpy as np
from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower import SO101FollowerConfig

# Config: which arm to use ("left" or "right")
ARM = "right"

# Sensitivity for phone tilt to joint movement (degrees per unit)
SENSITIVITY = 50.0

CONFIG_PATH = Path(__file__).parent.parent / "ports.toml"


def load_config() -> dict:
    with CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


def connect_phone() -> Any:
    """Connect to HEBI Mobile I/O app."""
    print("Looking for HEBI Mobile I/O app...")
    lookup = hebi.Lookup()
    time.sleep(2.0)

    group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
    if group is None:
        msg = "Mobile I/O not found. Make sure the app is open and on the same network."
        raise RuntimeError(msg)

    print("Connected to phone!")
    return group


def calibrate_phone(group: Any) -> tuple[np.ndarray, np.ndarray]:
    """Wait for B1 press to capture reference pose."""
    print("\nHold phone flat with screen facing up.")
    print("Press and hold B1 to set reference position...")

    while True:
        fbk = group.get_next_feedback(timeout_ms=100)
        if fbk is None:
            continue

        b1 = fbk.io.b.get_int(1)
        if hasattr(b1, "__iter__"):
            b1 = b1[0]

        if b1:
            ref_pos = np.array(fbk.ar_position).flatten()
            ref_quat = np.array(fbk.ar_orientation).flatten()
            print("Reference captured!")
            return ref_pos, ref_quat

        time.sleep(0.01)


def quat_to_euler(quat: np.ndarray) -> tuple[float, float, float]:
    """Convert quaternion (w,x,y,z) to euler angles (roll, pitch, yaw)."""
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def print_status(
    observation: dict,
    b1_pressed: bool,
    phone_euler: tuple[float, float, float],
    action: dict | None,
) -> None:
    """Print robot status, updating in place."""
    print("\033[2J\033[H", end="")
    print("=== Phone Teleop ===")
    print(f"Arm: {ARM}")
    print("-" * 55)
    print(f"B1: {'** ENABLED **' if b1_pressed else 'released (hold to control)'}")
    roll, pitch, yaw = phone_euler
    r, p, y = np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    print(f"Phone: roll={r:+6.1f}  pitch={p:+6.1f}  yaw={y:+6.1f}")
    print("-" * 55)
    print(f"{'Joint':<20} {'Current':>10} {'Target':>10}")
    print("-" * 55)

    for key in sorted(observation.keys()):
        val = observation[key]
        if isinstance(val, float):
            target_val = action.get(key, "") if action else ""
            if isinstance(target_val, float):
                print(f"{key:<20} {val:>10.2f} {target_val:>10.2f}")
            else:
                print(f"{key:<20} {val:>10.2f} {'--':>10}")


def main() -> None:
    config = load_config()
    follower_config = config["follower"][ARM]

    if not follower_config["port"]:
        print(f"Error: No port configured for {ARM} follower.")
        print("Please update ports.toml with the correct port.")
        return

    # Connect to phone first
    group = connect_phone()
    _ref_pos, ref_quat = calibrate_phone(group)
    ref_euler = quat_to_euler(ref_quat)

    # Setup robot
    print("\nConnecting to robot...")
    robot_config = SO101FollowerConfig(
        port=follower_config["port"],
        id=follower_config["id"],
    )
    robot = SO101Follower(robot_config)
    robot.connect(calibrate=False)

    # Capture initial joint positions as base
    initial_obs = robot.get_observation()
    base_joints = {k: v for k, v in initial_obs.items() if isinstance(v, float)}

    print("Starting teleop loop...")
    print("Hold B1 and tilt phone to move joints")
    print("Press Ctrl+C to exit\n")
    time.sleep(1)

    try:
        while True:
            fbk = group.get_next_feedback(timeout_ms=100)
            if fbk is None:
                continue

            # Read B1 button
            b1 = fbk.io.b.get_int(1)
            if hasattr(b1, "__iter__"):
                b1 = b1[0]
            b1_pressed = bool(b1)

            # Get phone orientation
            quat = np.array(fbk.ar_orientation).flatten()
            euler = quat_to_euler(quat)

            # Calculate delta from reference
            delta_roll = euler[0] - ref_euler[0]
            delta_pitch = euler[1] - ref_euler[1]
            delta_yaw = euler[2] - ref_euler[2]

            # Read A3 analog (gripper) - convert from -1..1 to 0..100
            a3 = 0.0
            if fbk.io.a.has_float(3):
                a3 = float(fbk.io.a.get_float(3))
            gripper_target = (a3 + 1.0) * 50.0

            # Get current robot state
            observation = robot.get_observation()

            action: dict | None = None
            if b1_pressed:
                # Map phone orientation to joint deltas
                action = {}

                # shoulder_pan: phone yaw (twist left/right)
                action["shoulder_pan.pos"] = (
                    base_joints["shoulder_pan.pos"] + np.degrees(delta_yaw) * SENSITIVITY / 50
                )

                # shoulder_lift: phone pitch (tilt forward/back)
                action["shoulder_lift.pos"] = (
                    base_joints["shoulder_lift.pos"] + np.degrees(delta_pitch) * SENSITIVITY / 50
                )

                # elbow_flex: also phone pitch
                action["elbow_flex.pos"] = (
                    base_joints["elbow_flex.pos"] - np.degrees(delta_pitch) * SENSITIVITY / 50
                )

                # wrist_flex: phone roll (tilt left/right)
                action["wrist_flex.pos"] = (
                    base_joints["wrist_flex.pos"] + np.degrees(delta_roll) * SENSITIVITY / 50
                )

                # wrist_roll: phone yaw
                action["wrist_roll.pos"] = (
                    base_joints["wrist_roll.pos"] + np.degrees(delta_yaw) * SENSITIVITY / 50
                )

                # gripper from A3
                action["gripper.pos"] = gripper_target

                robot.send_action(action)

            print_status(observation, b1_pressed, euler, action)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
