"""
Calibration utility for SO101 robot arms.

Calibrates all 4 bimanual arms (both leaders and both followers).
Calibrations are saved to the repo at calibration/{role}/{arm}.json

Usage:
    uv run setup/calibrate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer

from lib.config import get_calibration_dir
from lib.config import load_config
from lib.robots import get_single_follower
from lib.robots import get_single_leader

app = typer.Typer()


def validate_port(config: dict, role: str, arm: str) -> None:
    """Validate that a port is configured for the given role and arm."""
    port = config[role][arm].get("port")
    if not port:
        typer.echo(f"\nError: No port configured for {arm} {role}.")
        typer.echo("Please update config.toml with the correct port.")
        raise typer.Exit(1)


def create_arm_object(config: dict, role: str, arm: str):
    """Create an arm object with calibration stored in repo."""
    if role == "leader":
        return get_single_leader(config, arm)
    else:
        return get_single_follower(config, arm)


@app.command()
def main() -> None:
    """Calibrate all 4 bimanual robot arms (both leaders and both followers)."""
    config = load_config()

    typer.echo("\n=== Bimanual Robot Arm Calibration ===\n")

    # Validate all ports first
    for arm in ["left", "right"]:
        for role in ["leader", "follower"]:
            validate_port(config, role, arm)

    leader_cal_dir = get_calibration_dir("leader")
    follower_cal_dir = get_calibration_dir("follower")

    typer.echo("This will calibrate all 4 arms:")
    typer.echo(f"  Left Leader:    {config['leader']['left']['port']}")
    typer.echo(f"  Right Leader:   {config['leader']['right']['port']}")
    typer.echo(f"  Left Follower:  {config['follower']['left']['port']}")
    typer.echo(f"  Right Follower: {config['follower']['right']['port']}")
    typer.echo("\nCalibrations will be saved to:")
    typer.echo(f"  {leader_cal_dir}/bimanual_leader_left.json")
    typer.echo(f"  {leader_cal_dir}/bimanual_leader_right.json")
    typer.echo(f"  {follower_cal_dir}/bimanual_follower_left.json")
    typer.echo(f"  {follower_cal_dir}/bimanual_follower_right.json")

    # Calibrate leaders first
    typer.echo("\n--- Calibrating Leader Arms ---")

    for arm in ["left", "right"]:
        typer.echo(f"\nConnecting to {arm} leader arm...")
        arm_obj = create_arm_object(config, "leader", arm)
        arm_obj.connect(calibrate=False)

        typer.echo(f"Calibrating {arm} leader arm...")
        arm_obj.calibrate()
        typer.echo(f"{arm.capitalize()} leader calibration complete!")

        arm_obj.disconnect()

    # Calibrate followers
    typer.echo("\n--- Calibrating Follower Arms ---")

    for arm in ["left", "right"]:
        typer.echo(f"\nConnecting to {arm} follower arm...")
        arm_obj = create_arm_object(config, "follower", arm)
        arm_obj.connect(calibrate=False)

        typer.echo(f"Calibrating {arm} follower arm...")
        arm_obj.calibrate()
        typer.echo(f"{arm.capitalize()} follower calibration complete!")

        arm_obj.disconnect()

    typer.echo("\n=== Bimanual Calibration Complete! ===")
    typer.echo("All 4 arms have been calibrated successfully.")
    typer.echo("Calibration files are in the calibration/ directory - commit them to share.")


if __name__ == "__main__":
    app()
