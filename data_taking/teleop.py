"""
Bimanual teleoperation script using the BiSOLeader and BiSOFollower APIs.

Controls both left and right arm pairs simultaneously.

Usage:
    uv run data_taking/teleop.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.robots.bi_so_follower import BiSOFollower
from lerobot.robots.bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader
from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig

from lib.config import load_config
from lib.config import validate_config


def print_positions(leader_pos: dict, follower_pos: dict) -> None:
    """Print a table of leader and follower positions."""
    # Clear screen and move cursor to top
    print("\033[2J\033[H", end="")

    # Separate left and right joints
    left_keys = sorted(k for k in leader_pos if k.startswith("left_"))
    right_keys = sorted(k for k in leader_pos if k.startswith("right_"))

    # Header
    print(f"{'Joint':<25} {'Leader':>10} {'Follower':>10}")
    print("-" * 47)

    # Left arm
    print("LEFT ARM")
    for key in left_keys:
        short_key = key.removeprefix("left_")
        leader_val = leader_pos.get(key, 0.0)
        follower_val = follower_pos.get(key, 0.0)
        print(f"  {short_key:<23} {leader_val:>10.2f} {follower_val:>10.2f}")

    print()

    # Right arm
    print("RIGHT ARM")
    for key in right_keys:
        short_key = key.removeprefix("right_")
        leader_val = leader_pos.get(key, 0.0)
        follower_val = follower_pos.get(key, 0.0)
        print(f"  {short_key:<23} {leader_val:>10.2f} {follower_val:>10.2f}")

    print("\nPress Ctrl+C to exit")


def main() -> None:
    """Run bimanual teleoperation."""
    config = load_config()
    validate_config(config)

    # Get port configurations
    left_leader_cfg = config["leader"]["left"]
    right_leader_cfg = config["leader"]["right"]
    left_follower_cfg = config["follower"]["left"]
    right_follower_cfg = config["follower"]["right"]

    print("=== Bimanual Teleoperation ===\n")
    print(f"Left Leader:    {left_leader_cfg['port']}")
    print(f"Right Leader:   {right_leader_cfg['port']}")
    print(f"Left Follower:  {left_follower_cfg['port']}")
    print(f"Right Follower: {right_follower_cfg['port']}")

    # Create bimanual robot configuration
    robot_config = BiSOFollowerConfig(
        id="bimanual_follower",
        left_arm_config=SOFollowerConfig(
            port=left_follower_cfg["port"],
        ),
        right_arm_config=SOFollowerConfig(
            port=right_follower_cfg["port"],
        ),
    )

    # Create bimanual teleoperator configuration
    teleop_config = BiSOLeaderConfig(
        id="bimanual_leader",
        left_arm_config=SOLeaderConfig(
            port=left_leader_cfg["port"],
        ),
        right_arm_config=SOLeaderConfig(
            port=right_leader_cfg["port"],
        ),
    )

    # Create instances
    robot = BiSOFollower(robot_config)
    teleop = BiSOLeader(teleop_config)

    try:
        # Connect (skip calibration - assumes already calibrated)
        print("\nConnecting teleoperator...")
        teleop.connect(calibrate=False)
        print("Connecting robot...")
        robot.connect(calibrate=False)
        print("All devices connected!\n")

        # Teleoperation loop
        while True:
            # Get leader action (both arms combined)
            action = teleop.get_action()

            # Send to follower (both arms combined)
            robot.send_action(action)

            # Get follower observation (both arms combined)
            observation = robot.get_observation()

            # Display positions
            print_positions(action, observation)

    except KeyboardInterrupt:
        print("\n\nStopping teleoperation...")

    finally:
        if teleop.is_connected:
            print("Disconnecting teleoperator...")
            teleop.disconnect()
        if robot.is_connected:
            print("Disconnecting robot...")
            robot.disconnect()

    print("Done!")


if __name__ == "__main__":
    main()
