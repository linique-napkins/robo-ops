"""
Stow robot arms to their safe resting position.

Moves arms smoothly to the stow position using interpolated waypoints,
then releases torque so they can be moved by hand.

Usage:
    uv run utils/stow.py --leaders     # Stow leader arms
    uv run utils/stow.py --followers   # Stow follower arms
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer

from lib.config import load_config
from lib.config import validate_config
from lib.robots import get_bimanual_follower
from lib.robots import get_bimanual_leader
from lib.stow import stow_and_disconnect
from lib.stow import stow_leader

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(
    leaders: bool = typer.Option(False, "--leaders", "-l", help="Stow leader arms"),
    followers: bool = typer.Option(False, "--followers", "-f", help="Stow follower arms"),
) -> None:
    """Stow robot arms to their safe resting position."""
    if not leaders and not followers:
        typer.echo("Specify --leaders and/or --followers")
        raise typer.Exit(1)

    config = load_config()
    validate_config(config)

    if leaders:
        typer.echo("Connecting leader arms...")
        teleop = get_bimanual_leader(config)
        teleop.connect()
        stow_leader(teleop)
        teleop.disconnect()
        typer.echo("Leader arms disconnected.")

    if followers:
        typer.echo("Connecting follower arms...")
        robot = get_bimanual_follower(config)
        robot.connect()
        stow_and_disconnect(robot)


if __name__ == "__main__":
    app()
