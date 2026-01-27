"""
Inference script for running trained ACT policy on bimanual SO101 robot arms.

Loads a trained model from HuggingFace Hub and runs it on the robot in real-time.

Usage:
    uv run inference/run.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import typer
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.robots.bi_so_follower import BiSOFollower
from lerobot.robots.bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from lib.config import get_camera_config
from lib.config import load_config
from lib.config import validate_config

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
    """Extract inference configuration with defaults."""
    inference = config.get("inference", {})
    return {
        "policy_repo_id": inference.get("policy_repo_id", "jhimmens/linique-act"),
        "dataset_repo_id": inference.get("dataset_repo_id", "jhimmens/linique"),
        "device": inference.get("device", "mps"),
        "task": inference.get("task", "folding"),
        "fps": inference.get("fps", 30),
        "display": inference.get("display", True),
    }


def print_inference_config(config: dict, inference_cfg: dict) -> None:
    """Print inference configuration summary."""
    camera = get_camera_config(config)

    typer.echo("\nInference Configuration:")
    typer.echo(f"  Policy:         {inference_cfg['policy_repo_id']}")
    typer.echo(f"  Dataset:        {inference_cfg['dataset_repo_id']}")
    typer.echo(f"  Task:           {inference_cfg['task']}")
    typer.echo(f"  Device:         {inference_cfg['device']}")
    typer.echo(f"  FPS:            {inference_cfg['fps']}")
    typer.echo(f"  Display:        {inference_cfg['display']}")
    typer.echo("\nHardware Configuration:")
    typer.echo(f"  Left Follower:  {config['follower']['left']['port']}")
    typer.echo(f"  Right Follower: {config['follower']['right']['port']}")
    cam_info = f"{camera['path']} ({camera['width']}x{camera['height']} @ {camera['fps']}fps)"
    typer.echo(f"  Camera:         {cam_info}")


@app.command()
def main(
    display: bool = typer.Option(
        True,
        "--display/--no-display",
        help="Display visualization in Rerun",
    ),
) -> None:
    """Run trained ACT policy on bimanual SO101 robot arms."""
    typer.echo("\n=== ACT Policy Inference ===\n")

    # Load global config for hardware settings
    config = load_config()
    validate_config(config)

    # Load local config for inference settings
    local_config = load_config(LOCAL_CONFIG_PATH)
    config.update(local_config)

    # Get configurations
    inference_cfg = get_inference_config(config)
    camera_cfg = get_camera_config(config)

    # Override display from config if not specified
    if "display" in inference_cfg:
        display = inference_cfg["display"] and display

    print_inference_config(config, inference_cfg)

    if not typer.confirm("\nProceed with inference?"):
        typer.echo("Inference cancelled.")
        raise typer.Exit(0)

    # Setup device
    device = get_device(inference_cfg["device"])
    typer.echo(f"\nUsing device: {device}")

    # Load policy from HuggingFace Hub
    typer.echo(f"\nLoading policy from: {inference_cfg['policy_repo_id']}")
    policy_cfg = PreTrainedConfig.from_pretrained(inference_cfg["policy_repo_id"])
    policy_cfg.pretrained_path = inference_cfg["policy_repo_id"]

    # Load dataset for features (needed to build observation frames)
    typer.echo(f"Loading dataset metadata from: {inference_cfg['dataset_repo_id']}")
    dataset = LeRobotDataset(inference_cfg["dataset_repo_id"])

    # Create policy
    typer.echo("Creating policy model...")
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    policy.eval()
    policy.reset()
    policy.to(device)

    # Create processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=dataset.meta.stats,
    )

    # Create robot observation/action processors
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    # Get port configurations
    left_follower_cfg = config["follower"]["left"]
    right_follower_cfg = config["follower"]["right"]

    # Create bimanual robot configuration
    robot_config = BiSOFollowerConfig(
        id="bimanual_follower",
        left_arm_config=SOFollowerConfig(
            port=left_follower_cfg["port"],
            cameras={
                "top_cam": OpenCVCameraConfig(
                    index_or_path=camera_cfg["path"],
                    width=camera_cfg["width"],
                    height=camera_cfg["height"],
                    fps=camera_cfg["fps"],
                ),
            },
        ),
        right_arm_config=SOFollowerConfig(
            port=right_follower_cfg["port"],
        ),
    )

    # Initialize robot
    typer.echo("\nInitializing robot...")
    robot = BiSOFollower(robot_config)

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
        if display:
            init_rerun(session_name="inference")

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

        if robot.is_connected:
            typer.echo("Disconnecting robot...")
            robot.disconnect()

    total_time = time.time() - start_time
    typer.echo(f"\nInference complete. Ran for {total_time:.1f}s ({step} steps)")


if __name__ == "__main__":
    app()
