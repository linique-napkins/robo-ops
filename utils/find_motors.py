"""
Discover motor controller ports and identify which arm is on each.

Connects to each detected motor controller, reads positions, then asks you
to wiggle each arm so it can detect movement and map ports to arms.
Outputs stable /dev/serial/by-id/ paths for config.toml.

Usage:
    uv run utils/find_motors.py
"""

import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from lerobot.motors import Motor
from lerobot.motors import MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

app = typer.Typer()

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"

MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

ARM_ROLES = [
    ("leader", "left"),
    ("leader", "right"),
    ("follower", "left"),
    ("follower", "right"),
]


def get_by_id_path(port: str) -> str | None:
    """Resolve a /dev/ttyACM* port to its stable /dev/serial/by-id/ path."""
    by_id_dir = Path("/dev/serial/by-id")
    if not by_id_dir.exists():
        return None
    real_port = Path(port).resolve()
    for symlink in by_id_dir.iterdir():
        if symlink.resolve() == real_port:
            return str(symlink)
    return None


def find_motor_ports() -> list[str]:
    """Find all /dev/ttyACM* ports that have motor controllers."""
    ports = sorted(Path("/dev").glob("ttyACM*"))
    return [str(p) for p in ports]


def try_connect(port: str) -> FeetechMotorsBus | None:
    """Try to connect to a motor bus on a port."""
    try:
        bus = FeetechMotorsBus(port=port, motors=dict(MOTORS))
        bus.connect()
        return bus
    except Exception:
        return None


def read_positions(bus: FeetechMotorsBus) -> dict[str, int]:
    """Read raw positions from all motors."""
    return bus.sync_read("Present_Position", normalize=False)


def detect_movement(bus: FeetechMotorsBus, baseline: dict[str, int]) -> int:
    """Read positions and return total movement from baseline."""
    current = read_positions(bus)
    return sum(abs(current[k] - baseline[k]) for k in baseline)


def connect_all(ports: list[str]) -> dict[str, FeetechMotorsBus]:
    """Try connecting to motors on each port, return successful connections."""
    buses: dict[str, FeetechMotorsBus] = {}
    for port in ports:
        typer.echo(f"  Connecting to {port}...", nl=False)
        bus = try_connect(port)
        if bus:
            by_id = get_by_id_path(port)
            positions = read_positions(bus)
            pos_str = ", ".join(f"{v}" for v in positions.values())
            typer.echo(f" {GREEN}OK{RESET} positions=[{pos_str}]")
            if by_id:
                typer.echo(f"    {DIM}stable: {by_id}{RESET}")
            buses[port] = bus
        else:
            typer.echo(f" {YELLOW}no motors{RESET}")
    return buses


def identify_arms(
    buses: dict[str, FeetechMotorsBus],
) -> dict[str, tuple[str, str]]:
    """Identify which port belongs to which arm via movement detection."""
    assignments: dict[str, tuple[str, str]] = {}
    unassigned = list(buses.keys())

    for role, arm in ARM_ROLES:
        if not unassigned:
            typer.echo(f"{RED}No ports left for {role} {arm}.{RESET}")
            break

        if len(unassigned) == 1:
            port = unassigned.pop()
            assignments[port] = (role, arm)
            by_id = get_by_id_path(port) or port
            typer.echo(f"  {role} {arm}: {GREEN}{by_id}{RESET} (last remaining)")
            continue

        baselines = {p: read_positions(buses[p]) for p in unassigned}
        typer.echo(f"  Wiggle the {BOLD}{role.upper()} {arm.upper()}{RESET} arm, then press Enter.")
        input()

        movements = {p: detect_movement(buses[p], baselines[p]) for p in unassigned}
        best_port = max(movements, key=movements.get)
        best_movement = movements[best_port]

        if best_movement < 30:
            typer.echo(f"    {YELLOW}Warning: little movement detected ({best_movement}).{RESET}")

        assignments[best_port] = (role, arm)
        unassigned.remove(best_port)
        by_id = get_by_id_path(best_port) or best_port
        typer.echo(f"    -> {GREEN}{by_id}{RESET} (movement: {best_movement})\n")

    return assignments


@app.command()
def main() -> None:
    """Discover motor controller ports and identify arms by movement."""
    typer.echo(f"\n{BOLD}=== Motor Port Discovery ==={RESET}\n")

    ports = find_motor_ports()
    if not ports:
        typer.echo(f"{RED}No /dev/ttyACM* ports found. Are the motors plugged in?{RESET}")
        raise typer.Exit(1)

    typer.echo(f"Found {len(ports)} serial port(s): {', '.join(ports)}\n")

    buses = connect_all(ports)
    if not buses:
        typer.echo(f"\n{RED}No motor controllers found on any port.{RESET}")
        raise typer.Exit(1)

    typer.echo(f"\n{GREEN}Connected to {len(buses)} motor controller(s).{RESET}")
    if len(buses) < 4:
        typer.echo(
            f"{YELLOW}Expected 4 controllers (2 leader + 2 follower), found {len(buses)}.{RESET}"
        )

    typer.echo(f"\n{BOLD}--- Arm Identification ---{RESET}")
    typer.echo("Wiggle the requested arm when prompted. Keep others still.\n")

    assignments = identify_arms(buses)

    for bus in buses.values():
        with contextlib.suppress(Exception):
            bus.disconnect()

    typer.echo(f"\n{BOLD}=== Results ==={RESET}\n")
    typer.echo("Add these to your config.toml:\n")

    for port, (role, arm) in sorted(assignments.items(), key=lambda x: (x[1][0], x[1][1])):
        stable_path = get_by_id_path(port) or port
        typer.echo(f"[{role}.{arm}]")
        typer.echo(f'port = "{stable_path}"')
        typer.echo(f'id = "{role}_{arm}"')
        typer.echo()


if __name__ == "__main__":
    app()
