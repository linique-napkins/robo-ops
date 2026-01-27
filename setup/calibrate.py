"""
Calibration utility for SO101 robot arms.

Supports calibrating individual arms or all bimanual arms at once.

Usage:
    uv run setup/calibrate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from lerobot.robots.bi_so_follower import BiSOFollower
from lerobot.robots.bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader
from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SO101Leader
from lerobot.teleoperators.so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig

from lib.config import load_config

app = typer.Typer()


def validate_port(config: dict, role: str, arm: str) -> None:
    """Validate that a port is configured for the given role and arm."""
    port = config[role][arm].get("port")
    if not port:
        typer.echo(f"\nError: No port configured for {arm} {role}.")
        typer.echo("Please update config.toml with the correct port.")
        raise typer.Exit(1)


def get_calibration_mode() -> str:
    """Prompt user for calibration mode."""
    typer.echo("What would you like to calibrate?")
    typer.echo("  1. Single arm")
    typer.echo("  2. Bimanual (all 4 arms - both leaders and followers)")
    choice = typer.prompt("Enter your choice", type=int)
    if choice == 1:
        return "single"
    if choice == 2:
        return "bimanual"
    typer.echo("Invalid choice. Please enter 1 or 2.")
    raise typer.Exit(1)


def get_force_option() -> bool:
    """Prompt user whether to force overwrite existing calibration."""
    typer.echo("\nDo you want to overwrite existing calibration data?")
    typer.echo("  1. No, keep existing calibration if present (default)")
    typer.echo("  2. Yes, force recalibration even if already calibrated")
    choice = typer.prompt("Enter your choice", type=int, default=1)
    if choice == 1:
        return False
    if choice == 2:
        return True
    typer.echo("Invalid choice. Using default (no force).")
    return False


def get_arm_choice() -> str:
    """Prompt user for which arm to calibrate."""
    typer.echo("\nWhich arm do you want to calibrate?")
    typer.echo("  1. Left arm")
    typer.echo("  2. Right arm")
    choice = typer.prompt("Enter your choice", type=int)
    if choice == 1:
        return "left"
    if choice == 2:
        return "right"
    typer.echo("Invalid choice. Please enter 1 or 2.")
    raise typer.Exit(1)


def get_role_choice() -> str:
    """Prompt user for leader or follower."""
    typer.echo("\nIs this the leader (controller) or follower (robot) arm?")
    typer.echo("  1. Leader (the arm you move by hand)")
    typer.echo("  2. Follower (the robot arm that copies)")
    choice = typer.prompt("Enter your choice", type=int)
    if choice == 1:
        return "leader"
    if choice == 2:
        return "follower"
    typer.echo("Invalid choice. Please enter 1 or 2.")
    raise typer.Exit(1)


def calibrate_single_arm(config: dict, force: bool) -> None:
    """Calibrate a single arm."""
    arm = get_arm_choice()
    role = get_role_choice()

    validate_port(config, role, arm)

    arm_config = config[role][arm]
    port = arm_config["port"]
    arm_id = arm_config["id"]

    typer.echo(f"\nCalibrating {arm} {role} arm...")
    typer.echo(f"  Port: {port}")
    typer.echo(f"  ID: {arm_id}")

    if not typer.confirm("\nProceed with calibration?"):
        typer.echo("Calibration cancelled.")
        raise typer.Exit(0)

    if role == "leader":
        cfg = SO101LeaderConfig(port=port, id=arm_id)
        arm_obj = SO101Leader(cfg)
    else:
        cfg = SO101FollowerConfig(port=port, id=arm_id)
        arm_obj = SO101Follower(cfg)

    typer.echo("\nConnecting to arm...")
    arm_obj.connect(calibrate=False)

    if force:
        typer.echo("Force flag set - overwriting existing calibration...")
        arm_obj.calibration = {}

    typer.echo("Running calibration...")
    arm_obj.calibrate()

    typer.echo("Disconnecting...")
    arm_obj.disconnect()

    typer.echo("\nCalibration complete!")


def calibrate_bimanual(config: dict, force: bool) -> None:
    """Calibrate all bimanual arms (both leaders and both followers)."""
    for arm in ["left", "right"]:
        for role in ["leader", "follower"]:
            validate_port(config, role, arm)

    typer.echo("\nBimanual calibration will calibrate all 4 arms:")
    typer.echo(f"  Left Leader:    {config['leader']['left']['port']}")
    typer.echo(f"  Right Leader:   {config['leader']['right']['port']}")
    typer.echo(f"  Left Follower:  {config['follower']['left']['port']}")
    typer.echo(f"  Right Follower: {config['follower']['right']['port']}")

    if not typer.confirm("\nProceed with bimanual calibration?"):
        typer.echo("Calibration cancelled.")
        raise typer.Exit(0)

    # Create bimanual leader configuration
    leader_config = BiSOLeaderConfig(
        id="bimanual_leader",
        left_arm_config=SOLeaderConfig(
            port=config["leader"]["left"]["port"],
        ),
        right_arm_config=SOLeaderConfig(
            port=config["leader"]["right"]["port"],
        ),
    )

    # Create bimanual follower configuration
    follower_config = BiSOFollowerConfig(
        id="bimanual_follower",
        left_arm_config=SOFollowerConfig(
            port=config["follower"]["left"]["port"],
        ),
        right_arm_config=SOFollowerConfig(
            port=config["follower"]["right"]["port"],
        ),
    )

    # Calibrate leaders first
    typer.echo("\n--- Calibrating Leader Arms ---")
    typer.echo("Connecting to leader arms...")
    leader = BiSOLeader(leader_config)
    leader.connect(calibrate=False)

    if force:
        typer.echo("Force flag set - overwriting existing calibration...")
        leader.left_arm.calibration = {}
        leader.right_arm.calibration = {}

    typer.echo("\nCalibrating left leader arm...")
    leader.left_arm.calibrate()
    typer.echo("Left leader calibration complete!")

    typer.echo("\nCalibrating right leader arm...")
    leader.right_arm.calibrate()
    typer.echo("Right leader calibration complete!")

    typer.echo("Disconnecting leader arms...")
    leader.disconnect()

    # Calibrate followers
    typer.echo("\n--- Calibrating Follower Arms ---")
    typer.echo("Connecting to follower arms...")
    follower = BiSOFollower(follower_config)
    follower.connect(calibrate=False)

    if force:
        typer.echo("Force flag set - overwriting existing calibration...")
        follower.left_arm.calibration = {}
        follower.right_arm.calibration = {}

    typer.echo("\nCalibrating left follower arm...")
    follower.left_arm.calibrate()
    typer.echo("Left follower calibration complete!")

    typer.echo("\nCalibrating right follower arm...")
    follower.right_arm.calibrate()
    typer.echo("Right follower calibration complete!")

    typer.echo("Disconnecting follower arms...")
    follower.disconnect()

    typer.echo("\n=== Bimanual Calibration Complete! ===")
    typer.echo("All 4 arms have been calibrated successfully.")


@app.command()
def main() -> None:
    """Calibrate robot arms for teleoperation."""
    config = load_config()

    typer.echo("\n=== Robot Arm Calibration ===\n")

    mode = get_calibration_mode()
    force = get_force_option()

    if mode == "bimanual":
        calibrate_bimanual(config, force)
    else:
        calibrate_single_arm(config, force)


if __name__ == "__main__":
    app()
