"""
Remote teleoperation client for MacBook.

Reads leader arm positions via USB serial and sends them to the
demo server over WebSocket for real-time follower control.

Usage:
    uv run prod/teleop_client.py --server 100.x.y.z:8000
"""

import asyncio
import json
import sys
import time
import webbrowser
from pathlib import Path

import typer
import websockets
from lerobot.utils.robot_utils import precise_sleep

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import load_config
from lib.robots import get_bimanual_leader
from lib.stow import stow_leader

app = typer.Typer()


@app.command()
def main(
    server: str = typer.Option(..., "--server", "-s", help="Server address (host:port)"),
    fps: int = typer.Option(30, "--fps", help="Send rate in Hz"),
    open_browser: bool = typer.Option(
        True, "--browser/--no-browser", help="Open server UI in browser"
    ),
) -> None:
    """Send leader arm positions to the demo server via WebSocket."""
    config = load_config()
    teleop = get_bimanual_leader(config)

    typer.echo("Connecting leader arms...")
    teleop.connect(calibrate=False)
    typer.echo("Leader arms connected.")

    if open_browser:
        webbrowser.open(f"http://{server}/")

    try:
        asyncio.run(_send_loop(teleop, server, fps))
    except KeyboardInterrupt:
        typer.echo("\nStopping...")
    finally:
        stow_leader(teleop)
        if teleop.is_connected:
            teleop.disconnect()
        typer.echo("Done.")


async def _send_loop(teleop, server: str, fps: int) -> None:
    """Read leader arms and stream positions to the server over WebSocket."""
    uri = f"ws://{server}/ws/teleop"
    typer.echo(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        typer.echo("WebSocket connected. Sending leader positions at {fps}Hz...")
        step = 0
        while True:
            loop_start = time.perf_counter()

            action = teleop.get_action()
            positions = {k: float(v) for k, v in action.items()}
            await ws.send(json.dumps({"positions": positions}))

            step += 1
            if step % fps == 0:
                typer.echo(f"  Streaming... {step // fps}s")

            dt = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt, 0.0))


if __name__ == "__main__":
    app()
