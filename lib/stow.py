"""
Graceful robot stow on shutdown.

Moves follower arms to a safe stowed position before disconnecting,
preventing gravity-induced collapse onto the table.
"""

import time

from lerobot.utils.robot_utils import precise_sleep

# Joint names (without arm prefix)
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

STOW_POSITION = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -99.0,
    "elbow_flex": 99.0,
    "wrist_flex": -99.0,
    "wrist_roll": 0.0,
    "gripper": 80.0,
}
STOW_STEPS = 50
STOW_FPS = 30


def _is_bimanual(robot) -> bool:
    """Check if robot is a BiSOFollower (has left/right arm prefixes)."""
    obs = robot.get_observation()
    return any(k.startswith("left_") for k in obs)


def _build_action_keys(bimanual: bool) -> list[str]:
    """Build the full list of action keys for the robot."""
    if bimanual:
        return [f"{prefix}{joint}.pos" for prefix in ("left_", "right_") for joint in JOINT_NAMES]
    return [f"{joint}.pos" for joint in JOINT_NAMES]


def stow(robot) -> None:
    """Move robot arms to stow position without disconnecting.

    Works with both BiSOFollower (bimanual) and SOFollower (single arm).
    If stow fails, prints a warning and returns.

    Args:
        robot: Connected robot instance (BiSOFollower or SOFollower).
    """
    if not robot.is_connected:
        return

    try:
        bimanual = _is_bimanual(robot)
        action_keys = _build_action_keys(bimanual)

        # Read current positions
        obs = robot.get_observation()
        current = {}
        for key in action_keys:
            current[key] = obs.get(key, 0.0)

        # Build target positions
        target = {}
        for key in action_keys:
            # Strip prefix and .pos suffix to get bare joint name
            bare = key.replace("left_", "").replace("right_", "").replace(".pos", "")
            target[key] = STOW_POSITION[bare]

        print("Stowing robot...")

        # Interpolate from current to target
        for step in range(1, STOW_STEPS + 1):
            loop_start = time.perf_counter()
            alpha = step / STOW_STEPS

            waypoint = {}
            for key in action_keys:
                waypoint[key] = current[key] + alpha * (target[key] - current[key])

            robot.send_action(waypoint)

            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / STOW_FPS - dt_s, 0.0))

        print("Stow complete.")

    except Exception as e:
        print(f"Warning: stow failed ({e}).")


def stow_and_disconnect(robot) -> None:
    """Move robot arms to stow position, then disconnect.

    Args:
        robot: Connected robot instance (BiSOFollower or SOFollower).
    """
    stow(robot)
    if robot.is_connected:
        print("Disconnecting robot...")
        robot.disconnect()
