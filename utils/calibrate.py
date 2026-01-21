import tomllib
from pathlib import Path

import typer
from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader import SO101LeaderConfig

app = typer.Typer()

CONFIG_PATH = Path(__file__).parent.parent / "ports.toml"


def load_config() -> dict:
    with CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


def get_arm_choice() -> str:
    typer.echo("Which arm do you want to calibrate?")
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


@app.command()
def main(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing calibration"),
):
    """Calibrate a robot arm."""
    config = load_config()

    typer.echo("\n=== Robot Arm Calibration ===\n")

    arm = get_arm_choice()
    role = get_role_choice()

    arm_config = config[role][arm]
    port = arm_config["port"]
    arm_id = arm_config["id"]

    if not port:
        typer.echo(f"\nError: No port configured for {arm} {role}.")
        typer.echo("Please update ports.toml with the correct port.")
        raise typer.Exit(1)

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


if __name__ == "__main__":
    app()
