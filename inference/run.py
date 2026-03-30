"""
Inference script for running a trained policy on bimanual SO101 robot arms.

Loads a trained model (ACT, PI0, SmolVLA, etc.) and runs it on the robot in real-time.

Usage:
    uv run inference/run.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import typer
from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from lib.config import get_camera_config
from lib.config import get_urdf_config
from lib.config import load_config
from lib.config import validate_config
from lib.policy import load_policy_stack
from lib.robots import build_camera_configs
from lib.robots import get_bimanual_follower
from lib.stow import stow_and_disconnect
from lib.urdf_viz import init_rerun_with_urdf
from lib.urdf_viz import log_observation_and_action
from lib.urdf_viz import save_rrd
from utils.find_cameras import configure_exposure

app = typer.Typer()

LOCAL_CONFIG_PATH = Path(__file__).parent / "config.toml"


def get_device(device_str: str) -> torch.device:
    """Get torch device, validating availability."""
    if device_str == "cuda" and not torch.cuda.is_available():
        typer.echo("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    if device_str == "mps" and not torch.backends.mps.is_available():
        typer.echo("MPS not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def get_inference_config(config: dict) -> dict:
    """Extract inference configuration.

    Raises:
        KeyError: If [inference] section or required keys are missing.
    """
    inference = config["inference"]
    return {
        "policy_repo_id": inference["policy_repo_id"],
        "dataset_repo_id": inference["dataset_repo_id"],
        "device": inference["device"],
        "task": inference["task"],
        "fps": inference["fps"],
        "display": inference["display"],
        "temporal_ensemble_coeff": inference.get("temporal_ensemble_coeff"),
    }


def print_inference_config(config: dict, inference_cfg: dict) -> None:
    """Print inference configuration summary."""
    cameras = get_camera_config(config)

    typer.echo("\nInference Configuration:")
    typer.echo(f"  Policy:         {inference_cfg['policy_repo_id']}")
    typer.echo(f"  Dataset:        {inference_cfg['dataset_repo_id']}")
    typer.echo(f"  Task:           {inference_cfg['task']}")
    typer.echo(f"  Device:         {inference_cfg['device']}")
    typer.echo(f"  FPS:            {inference_cfg['fps']}")
    typer.echo(f"  Display:        {inference_cfg['display']}")
    te_coeff = inference_cfg["temporal_ensemble_coeff"]
    typer.echo(f"  Temporal Ens.:  {te_coeff if te_coeff is not None else 'disabled'}")
    typer.echo("\nHardware Configuration:")
    typer.echo(f"  Left Follower:  {config['follower']['left']['port']}")
    typer.echo(f"  Right Follower: {config['follower']['right']['port']}")
    typer.echo("  Cameras:")
    for name, cam in cameras.items():
        res = f"{cam['width']}x{cam['height']} @ {cam['fps']}fps"
        if cam["type"] == "realsense":
            typer.echo(f"    {name}: RealSense {cam['serial_number']} ({res})")
        else:
            typer.echo(f"    {name}: {cam['path']} ({res})")


@app.command()
def main(
    display: bool = typer.Option(
        True,
        "--display/--no-display",
        help="Display visualization in Rerun",
    ),
) -> None:
    """Run trained policy on bimanual SO101 robot arms."""
    typer.echo("\n=== Policy Inference ===\n")

    # Load global config for hardware settings
    config = load_config()
    validate_config(config)

    # Load local config for inference settings
    local_config = load_config(LOCAL_CONFIG_PATH)
    config.update(local_config)

    # Get configurations
    inference_cfg = get_inference_config(config)
    cameras_cfg = get_camera_config(config)

    # Override display from config if not specified
    if "display" in inference_cfg:
        display = inference_cfg["display"] and display

    print_inference_config(config, inference_cfg)

    # Apply camera exposure settings (v4l2 settings drift across reboots/reconnects)
    # Skip RealSense cameras — v4l2 controls don't apply to them
    typer.echo("\nApplying camera exposure settings...")
    for name, cam in cameras_cfg.items():
        if cam["type"] == "realsense":
            continue
        device_path = Path(cam["path"]).resolve()
        configure_exposure(str(device_path), name)

    # Setup device
    device = get_device(inference_cfg["device"])
    typer.echo(f"\nUsing device: {device}")

    # Load policy, processors, and dataset metadata via shared loader
    typer.echo(f"\nLoading policy from: {inference_cfg['policy_repo_id']}")
    (
        policy,
        policy_cfg,
        preprocessor,
        postprocessor,
        dataset,
        robot_action_processor,
        robot_observation_processor,
    ) = load_policy_stack(
        policy_repo_id=inference_cfg["policy_repo_id"],
        dataset_repo_id=inference_cfg["dataset_repo_id"],
        device=inference_cfg["device"],
        temporal_ensemble_coeff=inference_cfg["temporal_ensemble_coeff"],
    )
    typer.echo("Policy loaded.")

    # Build camera configs for robot
    camera_configs = build_camera_configs(cameras_cfg)

    # Initialize robot using factory function
    typer.echo("\nInitializing robot...")
    robot = get_bimanual_follower(config, cameras=camera_configs)

    listener = None
    fps = inference_cfg["fps"]
    task = inference_cfg["task"]
    step = 0
    start_time = time.time()

    try:
        # Connect robot
        typer.echo("Connecting robot...")
        robot.connect()

        # Initialize keyboard listener
        listener, events = init_keyboard_listener()

        # Initialize visualization
        visualizer = None
        if display:
            urdf_cfg = get_urdf_config(config)
            visualizer = init_rerun_with_urdf(
                session_name="inference",
                urdf_path=urdf_cfg["path"],
                left_offset=urdf_cfg["left_offset"],
                right_offset=urdf_cfg["right_offset"],
                left_rotation_deg=urdf_cfg["left_rotation"],
                right_rotation_deg=urdf_cfg["right_rotation"],
                camera_names=list(cameras_cfg.keys()),
            )

        log_say("Starting inference. Press Q to stop.", play_sounds=True)
        typer.echo("\n=== Running Inference ===")
        typer.echo("Controls:")
        typer.echo("  - Press 'q' to stop inference")
        typer.echo("")

        while not events["stop_recording"]:
            loop_start = time.perf_counter()

            # 1. Get observation from robot
            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)

            # 2. Build observation frame for policy
            observation_frame = build_dataset_frame(
                dataset.features,
                obs_processed,
                prefix=OBS_STR,
            )

            # 3. Run policy inference
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(inference_cfg["device"]),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=getattr(policy_cfg, "use_amp", False),
                task=task,
                robot_type=robot.name,
            )

            # 4. Convert to robot action format
            robot_action = make_robot_action(action_values, dataset.features)

            # 5. Send action to robot
            robot.send_action(robot_action_processor((robot_action, obs)))

            # 6. Log visualization if enabled
            if display:
                log_observation_and_action(
                    visualizer=visualizer,
                    observation=obs,
                    action=robot_action,
                    use_degrees=True,
                )

            step += 1

            # Log progress periodically
            if step % fps == 0:
                elapsed = time.time() - start_time
                typer.echo(f"Running... {elapsed:.1f}s elapsed, {step} steps")

            # Maintain target FPS
            loop_time = time.perf_counter() - loop_start
            sleep_time = max(1.0 / fps - loop_time, 0.0)
            precise_sleep(sleep_time)

        log_say("Inference stopped.", play_sounds=True)

    except KeyboardInterrupt:
        typer.echo("\n\nInference interrupted by user.")

    finally:
        if listener:
            listener.stop()

        stow_and_disconnect(robot)

        # Save Rerun recording
        if display and (rrd_path := save_rrd()):
            typer.echo(f"Rerun recording saved to: {rrd_path}")

    total_time = time.time() - start_time
    typer.echo(f"\nInference complete. Ran for {total_time:.1f}s ({step} steps)")


if __name__ == "__main__":
    app()
