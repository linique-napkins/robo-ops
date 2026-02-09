"""
Health check utility for SO101 robot arms.

Checks connection to all motors and reports voltage, temperature, load,
current, position, and error status for each joint in a color-coded table.

Usage:
    uv run setup/health.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer

from lib.config import load_config
from lib.robots import get_single_follower
from lib.robots import get_single_leader

app = typer.Typer()

# Joint names in order (matches motor IDs 1-6)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Ideal ranges for health metrics
IDEAL_RANGES = {
    "voltage": (6.5, 8.4),  # Volts (7.4V nominal LiPo)
    "temperature": (0, 45),  # Celsius
    "load": (-500, 500),  # Raw load units
    "current": (0, 500),  # mA
}

# Warning ranges (outside ideal but not critical)
WARNING_RANGES = {
    "voltage": (4.9, 8.0),
    "temperature": (0, 55),
    "load": (-800, 800),
    "current": (0, 800),
}


@dataclass
class MotorHealth:
    """Health data for a single motor."""

    name: str
    motor_id: int
    model: str
    firmware: str
    position: int
    velocity: int
    load: int
    voltage: float  # Volts
    temperature: int  # Celsius
    current: int  # mA
    moving: bool
    status: int
    torque_enabled: bool


def colorize(value: float | int, metric: str, fmt: str = "") -> str:
    """Colorize a value based on ideal/warning ranges."""
    ideal = IDEAL_RANGES.get(metric)
    warning = WARNING_RANGES.get(metric)

    formatted = fmt.format(value) if fmt else str(value)

    if ideal is None:
        return formatted

    min_ideal, max_ideal = ideal
    min_warn, max_warn = warning or ideal

    if min_ideal <= value <= max_ideal:
        return f"{GREEN}{formatted}{RESET}"
    elif min_warn <= value <= max_warn:
        return f"{YELLOW}{formatted}{RESET}"
    else:
        return f"{RED}{formatted}{RESET}"


def decode_status(status: int) -> tuple[str, str]:
    """Decode the status byte into error flags and color."""
    errors = []
    if status & 0x01:
        errors.append("Voltage")
    if status & 0x04:
        errors.append("Overheat")
    if status & 0x08:
        errors.append("Overload")
    if status & 0x20:
        errors.append("Torque")
    if status & 0x40:
        errors.append("Stall")

    if not errors:
        return f"{GREEN}OK{RESET}", "OK"
    else:
        return f"{RED}{','.join(errors)}{RESET}", ",".join(errors)


def read_motor_health(bus, motor_name: str) -> MotorHealth:
    """Read all health data from a single motor."""
    motor = bus.motors[motor_name]

    # Read all values (normalize=False to get raw values)
    position = bus.read("Present_Position", motor_name, normalize=False)
    velocity = bus.read("Present_Velocity", motor_name, normalize=False)
    load = bus.read("Present_Load", motor_name, normalize=False)
    voltage_raw = bus.read("Present_Voltage", motor_name, normalize=False)
    temperature = bus.read("Present_Temperature", motor_name, normalize=False)
    current = bus.read("Present_Current", motor_name, normalize=False)
    moving = bus.read("Moving", motor_name, normalize=False)
    status = bus.read("Status", motor_name, normalize=False)
    torque = bus.read("Torque_Enable", motor_name, normalize=False)

    # Read firmware version
    fw_major = bus.read("Firmware_Major_Version", motor_name, normalize=False)
    fw_minor = bus.read("Firmware_Minor_Version", motor_name, normalize=False)

    # Voltage is in units of 0.1V
    voltage = voltage_raw / 10.0

    return MotorHealth(
        name=motor_name,
        motor_id=motor.id,
        model=motor.model,
        firmware=f"{fw_major}.{fw_minor}",
        position=position,
        velocity=velocity,
        load=load,
        voltage=voltage,
        temperature=temperature,
        current=current,
        moving=bool(moving),
        status=status,
        torque_enabled=bool(torque),
    )


def print_table_header() -> None:
    """Print the table header."""
    typer.echo(
        f"\n{BOLD}{'Joint':<15} {'ID':>3} {'Volt':>7} {'Temp':>6} {'Load':>6} "
        f"{'Curr':>6} {'Pos':>6} {'Vel':>5} {'Torq':>5} {'Status':<12}{RESET}"
    )
    typer.echo("-" * 85)


def print_motor_row(health: MotorHealth) -> None:
    """Print a single motor row in the table."""
    status_colored, _ = decode_status(health.status)
    torque_str = f"{GREEN}ON{RESET}" if health.torque_enabled else f"{DIM}off{RESET}"

    volt_str = colorize(health.voltage, "voltage", "{:.1f}V")
    temp_str = colorize(health.temperature, "temperature", "{}°C")
    load_str = colorize(health.load, "load", "{:>5}")
    curr_str = colorize(health.current, "current", "{}mA")

    row = (
        f"{health.name:<15} {health.motor_id:>3} {volt_str:>16} {temp_str:>15} {load_str:>15} "
        f"{curr_str:>15} {health.position:>6} {health.velocity:>5} {torque_str:>14} "
        f"{status_colored}"
    )
    typer.echo(row)


def check_arm_health(config: dict, role: str, arm: str) -> list[MotorHealth]:
    """Check health of all motors in an arm."""
    port = config[role][arm].get("port")
    if not port:
        typer.echo(f"\n{YELLOW}No port configured for {arm} {role} - skipping{RESET}")
        return []

    typer.echo(f"\n{BOLD}=== {arm.upper()} {role.upper()} ({port}) ==={RESET}")
    print_table_header()

    try:
        if role == "leader":
            arm_obj = get_single_leader(config, arm)
        else:
            arm_obj = get_single_follower(config, arm)

        arm_obj.connect(calibrate=False)

        health_data = []
        for motor_name in JOINT_NAMES:
            try:
                health = read_motor_health(arm_obj.bus, motor_name)
                health_data.append(health)
                print_motor_row(health)
            except Exception as e:
                typer.echo(f"{RED}{motor_name:<15} ERROR: {e}{RESET}")

        arm_obj.disconnect()
        return health_data

    except Exception as e:
        typer.echo(f"{RED}Connection failed: {e}{RESET}")
        return []


def print_legend() -> None:
    """Print the color legend and ideal ranges."""
    typer.echo(f"\n{BOLD}Legend:{RESET}")
    typer.echo(f"  {GREEN}■{RESET} Good    {YELLOW}■{RESET} Warning    {RED}■{RESET} Critical")
    typer.echo(f"\n{BOLD}Ideal Ranges:{RESET}")
    typer.echo(
        f"  Voltage:     {IDEAL_RANGES['voltage'][0]}-{IDEAL_RANGES['voltage'][1]}V (7.4V nominal)"
    )
    typer.echo(
        f"  Temperature: {IDEAL_RANGES['temperature'][0]}-{IDEAL_RANGES['temperature'][1]}°C"
    )
    typer.echo(f"  Load:        {IDEAL_RANGES['load'][0]} to {IDEAL_RANGES['load'][1]} (signed)")
    typer.echo(f"  Current:     {IDEAL_RANGES['current'][0]}-{IDEAL_RANGES['current'][1]}mA")


def print_summary(all_health: dict[str, list[MotorHealth]]) -> None:
    """Print a summary of all motors."""
    # Collect all health data
    all_motors: list[tuple[str, MotorHealth]] = []
    for arm_name, motors in all_health.items():
        all_motors.extend((arm_name, motor) for motor in motors)

    if not all_motors:
        typer.echo(f"\n{RED}No motors connected.{RESET}")
        return

    # Find any issues
    issues = []
    for arm_name, motor in all_motors:
        _, status_raw = decode_status(motor.status)
        if status_raw != "OK":
            issues.append(f"{arm_name} {motor.name}: {status_raw}")
        if motor.temperature > WARNING_RANGES["temperature"][1]:
            issues.append(f"{arm_name} {motor.name}: High temp ({motor.temperature}°C)")
        if motor.voltage < WARNING_RANGES["voltage"][0]:
            issues.append(f"{arm_name} {motor.name}: Low voltage ({motor.voltage:.1f}V)")
        if motor.voltage > WARNING_RANGES["voltage"][1]:
            issues.append(f"{arm_name} {motor.name}: High voltage ({motor.voltage:.1f}V)")

    # Summary stats
    voltages = [m.voltage for _, m in all_motors]
    temps = [m.temperature for _, m in all_motors]

    typer.echo(f"\n{BOLD}=== SUMMARY ==={RESET}")
    typer.echo(f"Motors connected: {len(all_motors)}")
    typer.echo(f"Voltage range:    {min(voltages):.1f}V - {max(voltages):.1f}V")
    typer.echo(f"Temp range:       {min(temps)}°C - {max(temps)}°C")

    if issues:
        typer.echo(f"\n{RED}{BOLD}ISSUES DETECTED:{RESET}")
        for issue in issues:
            typer.echo(f"  {RED}✗{RESET} {issue}")
    else:
        typer.echo(f"\n{GREEN}{BOLD}✓ All motors healthy!{RESET}")


@app.command()
def main() -> None:
    """Check health of all bimanual robot arms."""
    config = load_config()

    typer.echo(f"\n{BOLD}{'=' * 40}")
    typer.echo("       ROBOT ARM HEALTH CHECK")
    typer.echo(f"{'=' * 40}{RESET}")

    all_health: dict[str, list[MotorHealth]] = {}

    # Check all arms
    for role in ["leader", "follower"]:
        for arm in ["left", "right"]:
            arm_key = f"{arm}_{role}"
            health = check_arm_health(config, role, arm)
            all_health[arm_key] = health

    print_summary(all_health)
    print_legend()


if __name__ == "__main__":
    app()
