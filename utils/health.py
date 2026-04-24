"""
Health check utility for SO101 robot arms and cameras.

Checks connection to all motors and reports voltage, temperature, load,
current, position, and error status for each joint in a color-coded table.

Also checks each configured camera: USB link speed (USB 2.0 vs 3.0+),
whether it connects, and whether a frame can be grabbed at the
configured resolution.

Usage:
    uv run utils/health.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import pyrealsense2 as rs
import typer

from lib.config import get_camera_config
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
class CameraHealth:
    """Health data for a single camera."""

    name: str
    cam_type: str
    identifier: str  # path for v4l, serial for realsense
    usb_speed_mbps: int | None  # raw speed in Mbps, None if unknown
    usb_label: str  # human-readable USB version
    connects: bool
    frame_ok: bool
    actual_width: int | None
    actual_height: int | None
    error: str = ""


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


# Ordered (min_mbps, template) pairs — first match wins.
_USB_SPEED_TIERS: list[tuple[int, str]] = [
    (20000, "USB 3.2 Gen 2x2 ({mbps} Mbps)"),
    (10000, "USB 3.1 Gen 2 ({mbps} Mbps)"),
    (5000, "USB 3.0 ({mbps} Mbps)"),
    (480, "USB 2.0 (480 Mbps)"),
    (12, "USB 1.1 ({mbps} Mbps)"),
    (0, "USB 1.0 ({mbps} Mbps)"),
]


def _usb_speed_label(mbps: int | None) -> str:
    """Convert USB link speed (Mbps) to a human-readable label."""
    if mbps is None:
        return "unknown"
    for threshold, template in _USB_SPEED_TIERS:
        if mbps >= threshold:
            return template.format(mbps=mbps)
    return f"USB ? ({mbps} Mbps)"


def _colorize_usb(label: str, mbps: int | None) -> str:
    """Green for USB 3+, yellow for USB 2.0, red for slower/unknown."""
    if mbps is None:
        return f"{RED}{label}{RESET}"
    if mbps >= 5000:
        return f"{GREEN}{label}{RESET}"
    if mbps >= 480:
        return f"{YELLOW}{label}{RESET}"
    return f"{RED}{label}{RESET}"


def get_v4l_usb_speed(path: str) -> int | None:
    """Get USB link speed (Mbps) for a /dev/video* or /dev/v4l/... device.

    Resolves the device path to /dev/videoN, then walks up from
    /sys/class/video4linux/videoN/device (the USB interface) one level
    to the USB device node and reads its `speed` attribute.
    """
    try:
        video_dev = Path(path).resolve()
        if not video_dev.name.startswith("video"):
            return None
        iface = Path(f"/sys/class/video4linux/{video_dev.name}/device").resolve()
        # iface is the USB interface dir (e.g. 1-7.2.3:1.0); parent is the USB device
        speed_file = iface.parent / "speed"
        if not speed_file.exists():
            return None
        return int(float(speed_file.read_text().strip()))
    except (OSError, ValueError):
        return None


def test_opencv_camera(cam_cfg: dict) -> tuple[bool, bool, int | None, int | None, str]:
    """Try to open an OpenCV camera and grab a frame.

    Returns (connects, frame_ok, actual_width, actual_height, error_message).
    """
    path = cam_cfg["path"]
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return False, False, None, None, "cv2.VideoCapture could not open device"

        if cam_cfg.get("fourcc"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam_cfg["fourcc"]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])

        ret, frame = cap.read()
        if not ret or frame is None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            return True, False, w, h, "connected but frame grab failed"

        h, w = frame.shape[:2]
        return True, True, w, h, ""
    except Exception as e:
        return False, False, None, None, str(e)
    finally:
        cap.release()


def test_realsense_camera(cam_cfg: dict) -> tuple[bool, bool, int | None, int | None, str, str]:
    """Try to query a RealSense camera by serial and grab a frame.

    Returns (connects, frame_ok, actual_width, actual_height, usb_label, error_message).
    USB label comes from the device's usb_type_descriptor (e.g. "3.2", "2.1").
    """
    serial = cam_cfg["serial_number"]
    try:
        ctx = rs.context()
        target = None
        for dev in ctx.query_devices():
            if dev.get_info(rs.camera_info.serial_number) == serial:
                target = dev
                break
        if target is None:
            return False, False, None, None, "unknown", f"serial {serial} not found"

        usb_desc = "unknown"
        if target.supports(rs.camera_info.usb_type_descriptor):
            usb_desc = f"USB {target.get_info(rs.camera_info.usb_type_descriptor)}"

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(
            rs.stream.color,
            cam_cfg["width"],
            cam_cfg["height"],
            rs.format.bgr8,
            cam_cfg["fps"],
        )
        try:
            pipeline.start(cfg)
        except Exception as e:
            return True, False, None, None, usb_desc, f"pipeline start failed: {e}"

        try:
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            color = frames.get_color_frame()
            if not color:
                return True, False, None, None, usb_desc, "no color frame"
            w = color.get_width()
            h = color.get_height()
            return True, True, w, h, usb_desc, ""
        finally:
            pipeline.stop()
    except Exception as e:
        return False, False, None, None, "unknown", str(e)


def _realsense_usb_mbps(usb_label: str) -> int | None:
    """Translate pyrealsense2 usb_type_descriptor (e.g. '3.2', '2.1') to Mbps."""
    token = usb_label.removeprefix("USB ").strip()
    try:
        major = int(token.split(".")[0])
    except (ValueError, IndexError):
        return None
    if major >= 3:
        return 5000
    if major == 2:
        return 480
    if major == 1:
        return 12
    return None


def check_camera_health(config: dict) -> list[CameraHealth]:
    """Check connection and USB speed for each configured camera."""
    try:
        cameras = get_camera_config(config)
    except KeyError:
        typer.echo(f"\n{YELLOW}No [camera] section in config - skipping cameras{RESET}")
        return []

    typer.echo(f"\n{BOLD}=== CAMERAS ==={RESET}")
    typer.echo(
        f"\n{BOLD}{'Name':<12} {'Type':<10} {'USB':<22} "
        f"{'Connect':<9} {'Frame':<6} {'Resolution':<13} {'Notes'}{RESET}"
    )
    typer.echo("-" * 100)

    results: list[CameraHealth] = []
    for name, cam_cfg in cameras.items():
        cam_type = cam_cfg.get("type", "opencv")

        if cam_type == "realsense":
            identifier = cam_cfg["serial_number"]
            connects, frame_ok, w, h, usb_label, err = test_realsense_camera(cam_cfg)
            mbps = _realsense_usb_mbps(usb_label) if connects else None
        else:
            identifier = cam_cfg["path"]
            mbps = get_v4l_usb_speed(identifier)
            usb_label = _usb_speed_label(mbps)
            connects, frame_ok, w, h, err = test_opencv_camera(cam_cfg)

        health = CameraHealth(
            name=name,
            cam_type=cam_type,
            identifier=identifier,
            usb_speed_mbps=mbps,
            usb_label=usb_label,
            connects=connects,
            frame_ok=frame_ok,
            actual_width=w,
            actual_height=h,
            error=err,
        )
        results.append(health)
        _print_camera_row(health, cam_cfg)

    return results


def _pad_colored(raw: str, width: int, colored: str) -> str:
    """Pad a colored string to `width` columns based on its raw (uncolored) length."""
    pad = max(0, width - len(raw))
    return colored + " " * pad


def _print_camera_row(health: CameraHealth, cam_cfg: dict) -> None:
    """Print a single camera row."""
    usb_col = _pad_colored(
        health.usb_label, 22, _colorize_usb(health.usb_label, health.usb_speed_mbps)
    )

    connect_raw = "yes" if health.connects else "NO"
    connect_col = _pad_colored(
        connect_raw, 9, f"{GREEN if health.connects else RED}{connect_raw}{RESET}"
    )

    if health.connects and health.frame_ok:
        frame_raw, frame_colored = "ok", f"{GREEN}ok{RESET}"
    elif health.connects:
        frame_raw, frame_colored = "fail", f"{RED}fail{RESET}"
    else:
        frame_raw, frame_colored = "-", f"{DIM}-{RESET}"
    frame_col = _pad_colored(frame_raw, 6, frame_colored)

    if health.actual_width and health.actual_height:
        want_w, want_h = cam_cfg["width"], cam_cfg["height"]
        actual = f"{health.actual_width}x{health.actual_height}"
        if (health.actual_width, health.actual_height) != (want_w, want_h):
            res_raw = f"{actual} (want {want_w}x{want_h})"
            res_col = f"{YELLOW}{res_raw}{RESET}"
        else:
            res_raw, res_col = actual, f"{GREEN}{actual}{RESET}"
    else:
        res_raw, res_col = "-", f"{DIM}-{RESET}"
    res_col = _pad_colored(res_raw, 23, res_col)

    notes = health.error
    typer.echo(
        f"{health.name:<12} {health.cam_type:<10} {usb_col} {connect_col} "
        f"{frame_col} {res_col} {notes}"
    )


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


def _collect_motor_issues(all_motors: list[tuple[str, MotorHealth]]) -> list[str]:
    """Return a list of issue strings for all motors."""
    issues: list[str] = []
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
    return issues


def _collect_camera_issues(camera_health: list[CameraHealth]) -> list[str]:
    """Return a list of issue strings for all cameras."""
    issues: list[str] = []
    for cam in camera_health:
        if not cam.connects:
            issues.append(f"camera {cam.name}: connection failed ({cam.error})")
        elif not cam.frame_ok:
            issues.append(f"camera {cam.name}: no frame ({cam.error})")
        elif cam.usb_speed_mbps is not None and cam.usb_speed_mbps < 480:
            issues.append(f"camera {cam.name}: slow USB link ({cam.usb_label})")
    return issues


def print_summary(
    all_health: dict[str, list[MotorHealth]],
    camera_health: list[CameraHealth] | None = None,
) -> None:
    """Print a summary of all motors and cameras."""
    all_motors: list[tuple[str, MotorHealth]] = []
    for arm_name, motors in all_health.items():
        all_motors.extend((arm_name, motor) for motor in motors)

    typer.echo(f"\n{BOLD}=== SUMMARY ==={RESET}")

    if not all_motors:
        typer.echo(f"{RED}No motors connected.{RESET}")
    else:
        voltages = [m.voltage for _, m in all_motors]
        temps = [m.temperature for _, m in all_motors]
        typer.echo(f"Motors connected: {len(all_motors)}")
        typer.echo(f"Voltage range:    {min(voltages):.1f}V - {max(voltages):.1f}V")
        typer.echo(f"Temp range:       {min(temps)}°C - {max(temps)}°C")

    if camera_health:
        ok_count = sum(1 for c in camera_health if c.connects and c.frame_ok)
        typer.echo(
            f"Cameras ok:       {ok_count}/{len(camera_health)} (connected + frame grab succeeded)"
        )

    issues = _collect_motor_issues(all_motors)
    if camera_health:
        issues.extend(_collect_camera_issues(camera_health))

    if issues:
        typer.echo(f"\n{RED}{BOLD}ISSUES DETECTED:{RESET}")
        for issue in issues:
            typer.echo(f"  {RED}✗{RESET} {issue}")
    else:
        typer.echo(f"\n{GREEN}{BOLD}✓ All hardware healthy!{RESET}")


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

    camera_health = check_camera_health(config)

    print_summary(all_health, camera_health)
    print_legend()


if __name__ == "__main__":
    app()
