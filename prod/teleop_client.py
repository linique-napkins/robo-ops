"""
Remote teleoperation client for MacBook.

Reads leader arm positions via USB serial and sends them to the
demo server over WebSocket for real-time follower control.

Usage:
    uv run prod/teleop_client.py --server 100.x.y.z:8000
"""

import asyncio
import json
import ssl
import sys
import time
import urllib.request
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

# Tailscale certs may not be in the system trust store, so we skip verification
# for the private tailnet. Traffic is already encrypted and authenticated by Tailscale.
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE


def _api_request(server: str, path: str, method: str = "POST") -> dict:
    """Make an HTTP request to a server API endpoint and return the JSON response."""
    url = f"https://{server}{path}"
    data = b"" if method == "POST" else None
    req = urllib.request.Request(url, method=method, data=data)
    try:
        with urllib.request.urlopen(req, context=_ssl_ctx) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        return body


def _ensure_server_ready(server: str) -> None:
    """Ensure the server robot is connected and in teleop mode."""
    # Check current state
    state = _api_request(server, "/api/state", method="GET")
    current = state.get("state", "disconnected")

    # Connect robot if needed
    if current == "disconnected":
        typer.echo("Server robot disconnected — connecting...")
        resp = _api_request(server, "/api/connect")
        if not resp.get("ok"):
            typer.echo(f"Server failed to connect robot: {resp.get('error', 'unknown')}")
            raise typer.Exit(1)
        typer.echo("Server robot connected.")
        current = "idle"

    # Stop any existing operation to get back to idle
    if current not in ("idle", "teleop"):
        typer.echo(f"Server is {current} — stopping current operation...")
        _api_request(server, "/api/teleop/stop")
        current = "idle"

    # Start teleop if not already active
    if current != "teleop":
        resp = _api_request(server, "/api/teleop/start")
        if not resp.get("ok"):
            typer.echo(f"Server rejected teleop start: {resp.get('error', 'unknown')}")
            raise typer.Exit(1)


@app.command()
def main(
    server: str = typer.Option("nvd-compute.tail6bd9d.ts.net:8000", "--server", "-s", help="Server address (host:port)"),
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
        webbrowser.open(f"https://{server}/")

    # Ensure server robot is connected and in teleop mode
    typer.echo("Starting teleop mode on server...")
    try:
        _ensure_server_ready(server)
    except urllib.error.URLError as e:
        typer.echo(f"Failed to reach server: {e}")
        raise typer.Exit(1)
    typer.echo("Server teleop mode active.")

    try:
        asyncio.run(_send_loop(teleop, server, fps))
    except KeyboardInterrupt:
        typer.echo("\nStopping...")
    finally:
        # Stop teleop mode on the server
        try:
            _api_request(server, "/api/teleop/stop")
            typer.echo("Server teleop mode stopped.")
        except Exception:
            typer.echo("Warning: could not stop teleop on server.")
        stow_leader(teleop)
        if teleop.is_connected:
            teleop.disconnect()
        typer.echo("Done.")


async def _send_loop(teleop, server: str, fps: int) -> None:
    """Read leader arms and stream positions to the server over WebSocket.

    Auto-reconnects on connection loss with exponential backoff.
    """
    uri = f"wss://{server}/ws/teleop"
    backoff = 1.0

    while True:
        try:
            typer.echo(f"Connecting to {uri}...")
            async with websockets.connect(uri, ssl=_ssl_ctx) as ws:
                typer.echo(f"WebSocket connected. Sending leader positions at {fps}Hz...")
                backoff = 1.0
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
        except (websockets.ConnectionClosed, OSError) as e:
            typer.echo(f"Connection lost: {e}. Reconnecting in {backoff:.0f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)


if __name__ == "__main__":
    app()
