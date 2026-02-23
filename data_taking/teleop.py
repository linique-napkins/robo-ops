"""
Bimanual teleoperation script using the BiSOLeader and BiSOFollower APIs.

Controls both left and right arm pairs simultaneously with optional camera visualization.

Usage:
    uv run data_taking/teleop.py
    uv run data_taking/teleop.py --display  # With Rerun + URDF + camera visualization
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.robot_utils import precise_sleep

from lib.config import get_camera_config
from lib.config import get_urdf_config
from lib.config import load_config
from lib.config import validate_config
from lib.robots import get_bimanual_follower
from lib.robots import get_bimanual_leader
from lib.stow import stow_and_disconnect
from lib.urdf_viz import init_rerun_with_urdf
from lib.urdf_viz import log_observation_and_action
from lib.urdf_viz import save_rrd

# Joint display order (short names without arm prefix)
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
NUM_JOINT_LINES = len(JOINT_ORDER)  # Lines to move cursor up (just the joint rows)


def move_cursor_up(lines: int) -> None:
    """Move the cursor up by a specified number of lines."""
    print(f"\033[{lines}A", end="", flush=True)


def print_table_header() -> None:
    """Print the table header (only once at start)."""
    print()
    print("-" * 70)
    print(f"{'JOINT':<20} | {'LEFT':^22} | {'RIGHT':^22}")
    print(f"{'':<20} | {'Leader':>10} {'Follower':>10} | {'Leader':>10} {'Follower':>10}")
    print("-" * 70)


def print_positions(leader_pos: dict, follower_pos: dict) -> None:
    """Print the joint position rows (overwrites previous values in place)."""
    for joint in JOINT_ORDER:
        left_key = f"left_{joint}.pos"
        right_key = f"right_{joint}.pos"

        left_leader = leader_pos.get(left_key, 0.0)
        left_follower = follower_pos.get(left_key, 0.0)
        right_leader = leader_pos.get(right_key, 0.0)
        right_follower = follower_pos.get(right_key, 0.0)

        # Use \033[K to clear to end of line (removes any stale characters)
        print(
            f"{joint:<20} | {left_leader:>10.2f} {left_follower:>10.2f} "
            f"| {right_leader:>10.2f} {right_follower:>10.2f}\033[K"
        )


def main() -> None:  # noqa: PLR0912
    """Run bimanual teleoperation."""
    parser = argparse.ArgumentParser(
        description="Bimanual teleoperation with optional visualization"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Enable Rerun visualization with 3D robot model and cameras",
    )
    args = parser.parse_args()

    config = load_config()
    validate_config(config)

    print("=== Bimanual Teleoperation ===\n")
    print(f"Left Leader:    {config['leader']['left']['port']}")
    print(f"Right Leader:   {config['leader']['right']['port']}")
    print(f"Left Follower:  {config['follower']['left']['port']}")
    print(f"Right Follower: {config['follower']['right']['port']}")

    # Build camera configs if display is enabled
    camera_configs = {}
    cameras_cfg = {}
    if args.display:
        cameras_cfg = get_camera_config(config)
        print("  Cameras:")
        for name, cam in cameras_cfg.items():
            print(f"    {name}: {cam['path']} ({cam['width']}x{cam['height']} @ {cam['fps']}fps)")
            camera_configs[f"{name}_cam"] = OpenCVCameraConfig(
                index_or_path=cam["path"],
                width=cam["width"],
                height=cam["height"],
                fps=cam["fps"],
                fourcc=cam["fourcc"],
            )

    # Create robot and teleoperator using factory functions
    robot = get_bimanual_follower(config, cameras=camera_configs if camera_configs else None)
    teleop = get_bimanual_leader(config)

    # Initialize visualization if requested
    visualizer = None
    if args.display:
        urdf_cfg = get_urdf_config(config)
        print("\nInitializing Rerun visualization with URDF...")
        visualizer = init_rerun_with_urdf(
            session_name="teleop",
            urdf_path=urdf_cfg["path"],
            left_offset=urdf_cfg["left_offset"],
            right_offset=urdf_cfg["right_offset"],
            left_rotation_deg=urdf_cfg["left_rotation"],
            right_rotation_deg=urdf_cfg["right_rotation"],
            camera_names=list(cameras_cfg.keys()) if cameras_cfg else None,
        )
        if visualizer:
            print("URDF visualization initialized!")
        else:
            print("Warning: URDF visualization failed to initialize")

    try:
        # Connect (skip calibration - assumes already calibrated)
        print("\nConnecting teleoperator...")
        teleop.connect(calibrate=False)
        print("Connecting robot...")
        robot.connect(calibrate=False)
        print("All devices connected!")
        print("\nPress Ctrl+C to exit")

        fps = 30  # Target loop rate

        # Print header once (only in terminal mode)
        if not args.display:
            print_table_header()

        # Teleoperation loop
        while True:
            loop_start = time.perf_counter()

            # Get leader action (both arms combined)
            action = teleop.get_action()

            # Send to follower (both arms combined)
            robot.send_action(action)

            # Get follower observation (both arms combined + camera images)
            observation = robot.get_observation()

            # Display positions in terminal or Rerun
            if not args.display:
                print_positions(action, observation)
                # Move cursor up to overwrite on next iteration
                move_cursor_up(NUM_JOINT_LINES)
            else:
                # Log to Rerun (includes URDF joint updates and camera images)
                log_observation_and_action(
                    visualizer=visualizer,
                    observation=observation,
                    action=action,
                    use_degrees=True,
                )

            # Maintain target FPS
            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0.0))

    except KeyboardInterrupt:
        # Move cursor down past the table before printing exit message
        if not args.display:
            print()  # Move past the joint rows
        print("\nStopping teleoperation...")

    finally:
        if teleop.is_connected:
            print("Disconnecting teleoperator...")
            teleop.disconnect()
        stow_and_disconnect(robot)

        # Save Rerun recording
        if args.display:
            rrd_path = save_rrd()
            if rrd_path:
                print(f"Rerun recording saved to: {rrd_path}")

    print("Done!")


if __name__ == "__main__":
    main()
