"""
Visualize color, depth, and IMU from a connected Intel RealSense camera.

Shows color and depth feeds side by side with device info overlay and
live IMU readings (accelerometer + gyroscope) for D435I models.

Usage:
    uv run utils/test_realsense.py
    uv run utils/test_realsense.py --serial 0123456789
"""

import argparse
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def get_device_info(device: rs.device) -> dict[str, str]:
    """Extract all available info from a RealSense device."""
    info = {}
    labels = ["name", "serial_number", "firmware_version", "physical_port",
              "product_id", "product_line", "usb_type_descriptor"]
    fields = [getattr(rs.camera_info, l) for l in labels]
    for label, field in zip(labels, fields):
        try:
            info[label] = device.get_info(field)
        except RuntimeError:
            pass
    return info


def get_sensor_info(device: rs.device) -> list[dict]:
    """Get info about each sensor and its supported stream profiles."""
    sensors = []
    for sensor in device.query_sensors():
        s = {"name": sensor.get_info(rs.camera_info.name), "profiles": []}
        for profile in sensor.get_stream_profiles():
            if profile.is_video_stream_profile():
                vp = profile.as_video_stream_profile()
                s["profiles"].append({
                    "stream": vp.stream_name(),
                    "format": vp.format().name,
                    "w": vp.width(),
                    "h": vp.height(),
                    "fps": vp.fps(),
                    "default": profile.is_default(),
                })
        sensors.append(s)
    return sensors


def draw_text(img: np.ndarray, lines: list[str], origin: tuple[int, int] = (10, 20),
              scale: float = 0.5, color: tuple = (255, 255, 255), thickness: int = 1,
              line_gap: int = 20, bg: bool = True) -> None:
    """Draw multiple lines of text with optional dark background."""
    x, y = origin
    for i, line in enumerate(lines):
        pos = (x, y + i * line_gap)
        if bg:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            cv2.rectangle(img, (pos[0] - 2, pos[1] - th - 4), (pos[0] + tw + 2, pos[1] + 4), (0, 0, 0), -1)
        cv2.putText(img, line, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def has_imu(device: rs.device) -> bool:
    """Check if the device has a motion module (IMU)."""
    for sensor in device.query_sensors():
        if sensor.is_motion_sensor():
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="RealSense camera viewer")
    parser.add_argument("--serial", type=str, default=None, help="Device serial number (uses first device if omitted)")
    parser.add_argument("--imu", action="store_true", help="Enable IMU streams (needs udev rules)")
    args = parser.parse_args()

    # --- Discover devices ---
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("ERROR: No RealSense devices found.")
        sys.exit(1)

    print(f"Found {len(devices)} RealSense device(s):\n")

    target_device = None
    for dev in devices:
        sn = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"  [{sn}] {name}")
        if args.serial is None or sn == args.serial:
            target_device = dev

    if target_device is None:
        print(f"\nERROR: No device found with serial '{args.serial}'")
        sys.exit(1)

    serial = target_device.get_info(rs.camera_info.serial_number)
    print(f"\nUsing device: {serial}")

    # --- Print full device info ---
    dev_info = get_device_info(target_device)
    print("\n=== Device Info ===")
    for k, v in dev_info.items():
        print(f"  {k}: {v}")

    # --- Print sensor / stream info ---
    sensor_info = get_sensor_info(target_device)
    print("\n=== Sensors & Default Profiles ===")
    for s in sensor_info:
        print(f"  Sensor: {s['name']}")
        defaults = [p for p in s["profiles"] if p["default"]]
        for p in defaults:
            print(f"    {p['stream']} {p['w']}x{p['h']} @ {p['fps']}fps ({p['format']})")

    # --- Configure and start pipeline ---
    usb_type = dev_info.get("usb_type_descriptor", "3")
    if usb_type.startswith("2"):
        print(f"\nERROR: RealSense D435I is on USB {usb_type} — cannot stream over USB 2.x.")
        print("The D435I requires USB 3.0+ for all video (including RGB-only).")
        print("Use a USB 3.0 cable and plug into a USB 3.0 port (blue insert / marked 'SS').")
        sys.exit(1)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # IMU needs udev rules on Linux — check permissions before enabling
    imu_enabled = False
    if has_imu(target_device) and args.imu:
        try:
            test_cfg = rs.config()
            test_cfg.enable_device(serial)
            test_cfg.enable_stream(rs.stream.accel)
            test_cfg.enable_stream(rs.stream.gyro)
            test_pipe = rs.pipeline()
            test_profile = test_pipe.start(test_cfg)
            test_pipe.stop()
            imu_enabled = True
            config.enable_stream(rs.stream.accel)
            config.enable_stream(rs.stream.gyro)
        except RuntimeError:
            print("WARN: IMU found but access denied. Run with sudo or install udev rules.")

    profile = pipeline.start(config)

    # Read back actual stream settings
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Print intrinsics
    color_intrinsics = color_profile.get_intrinsics()
    depth_intrinsics = depth_profile.get_intrinsics()

    print(f"\n=== Active Streams ===")
    print(f"  Color: {color_profile.width()}x{color_profile.height()} @ {color_profile.fps()}fps "
          f"({color_profile.format().name})")
    print(f"  Depth: {depth_profile.width()}x{depth_profile.height()} @ {depth_profile.fps()}fps "
          f"({depth_profile.format().name})")
    print(f"  Depth scale: {depth_scale:.6f} m/unit")
    if imu_enabled:
        print(f"  IMU: accelerometer + gyroscope enabled")

    print(f"\n=== Intrinsics ===")
    print(f"  Color: fx={color_intrinsics.fx:.1f} fy={color_intrinsics.fy:.1f} "
          f"cx={color_intrinsics.ppx:.1f} cy={color_intrinsics.ppy:.1f} "
          f"model={color_intrinsics.model}")
    print(f"  Depth: fx={depth_intrinsics.fx:.1f} fy={depth_intrinsics.fy:.1f} "
          f"cx={depth_intrinsics.ppx:.1f} cy={depth_intrinsics.ppy:.1f} "
          f"model={depth_intrinsics.model}")

    print(f"\n  Depth sensor options:")
    for opt in [rs.option.min_distance, rs.option.visual_preset, rs.option.laser_power,
                rs.option.emitter_enabled, rs.option.exposure, rs.option.gain]:
        try:
            val = depth_sensor.get_option(opt)
            rng = depth_sensor.get_option_range(opt)
            print(f"    {opt.name}: {val} (range: {rng.min}-{rng.max})")
        except RuntimeError:
            pass

    colorizer = rs.colorizer()

    print("\nStreaming... Press 'q' to quit.\n")

    # --- Info lines for overlay ---
    info_lines = [
        f"{dev_info.get('name', '?')} | SN: {serial}",
        f"FW: {dev_info.get('firmware_version', '?')} | USB {usb_type}"
        + (" (USB2 limits FPS!)" if usb_type.startswith("2") else ""),
        f"Depth scale: {depth_scale:.6f} m/unit",
    ]

    frame_count = 0
    t_start = time.time()
    fps_display = 0.0
    accel_data = (0.0, 0.0, 0.0)
    gyro_data = (0.0, 0.0, 0.0)

    try:
        while True:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # IMU data (arrives at higher rate, grab latest)
            if imu_enabled:
                accel_frame = frames.first_or_default(rs.stream.accel)
                gyro_frame = frames.first_or_default(rs.stream.gyro)
                if accel_frame:
                    a = accel_frame.as_motion_frame().get_motion_data()
                    accel_data = (a.x, a.y, a.z)
                if gyro_frame:
                    g = gyro_frame.as_motion_frame().get_motion_data()
                    gyro_data = (g.x, g.y, g.z)

            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            # Color image (RGB -> BGR for OpenCV)
            color_img = np.asanyarray(color_frame.get_data())
            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            # Depth colorized
            depth_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # Raw depth for stats
            depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float64) * depth_scale
            valid = depth_raw[depth_raw > 0]
            if len(valid) > 0:
                d_min, d_max, d_mean = valid.min(), valid.max(), valid.mean()
            else:
                d_min = d_max = d_mean = 0.0

            # Resize depth to match color dimensions
            ch, cw = color_bgr.shape[:2]
            depth_display = cv2.resize(depth_colorized, (cw, ch))

            # Overlay info on color
            overlay_lines = info_lines + [
                f"FPS: {fps_display:.1f}",
                f"Depth: {d_min:.2f} - {d_max:.2f} m (mean {d_mean:.2f})",
            ]
            if imu_enabled:
                overlay_lines += [
                    f"Accel: x={accel_data[0]:+.2f} y={accel_data[1]:+.2f} z={accel_data[2]:+.2f} m/s2",
                    f"Gyro:  x={gyro_data[0]:+.3f} y={gyro_data[1]:+.3f} z={gyro_data[2]:+.3f} rad/s",
                ]
            draw_text(color_bgr, overlay_lines)

            # Label panels
            draw_text(color_bgr, ["Color"], (cw - 60, 20), scale=0.5, color=(0, 255, 0))
            draw_text(depth_display, ["Depth"], (cw - 60, 20), scale=0.5, color=(0, 255, 0))

            canvas = np.hstack([color_bgr, depth_display])

            # Scale down if too large for the screen
            max_w = 1920
            if canvas.shape[1] > max_w:
                s = max_w / canvas.shape[1]
                canvas = cv2.resize(canvas, None, fx=s, fy=s)

            cv2.imshow("RealSense Viewer", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
