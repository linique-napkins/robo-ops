"""
Inference script for running a trained policy on bimanual SO101 robot arms.

Loads a trained model (ACT, PI0, SmolVLA, etc.) and runs it on the robot in real-time.

Usage:
    uv run inference/run.py
"""

import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import typer
from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import ACTION
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import TimerManager
from lerobot.utils.utils import log_say

from lib.config import get_camera_config
from lib.config import get_git_info
from lib.config import get_local_dataset_path
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

INFERENCE_DIR = Path(__file__).parent
DEFAULT_CONFIG = "config.toml"


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
        "n_action_steps": inference.get("n_action_steps"),
        "record": inference.get("record", False),
        "record_repo_id": inference.get("record_repo_id"),
        "rename_map": inference.get("rename_map", {}),
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
    typer.echo(f"  Recording:      {inference_cfg['record']}")
    if inference_cfg["record_repo_id"]:
        typer.echo(f"  Record Repo:    {inference_cfg['record_repo_id']}")
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
def main(  # noqa: PLR0912
    config_file: str = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="Inference config file name in inference/ directory",
    ),
    display: bool = typer.Option(
        True,
        "--display/--no-display",
        help="Display visualization in Rerun",
    ),
    record: bool = typer.Option(
        False,
        "--record/--no-record",
        help="Record inference data as a LeRobot dataset",
    ),
    push_to_hub: bool = typer.Option(
        False,
        "--push/--no-push",
        help="Push recorded dataset to Hugging Face Hub",
    ),
) -> None:
    """Run trained policy on bimanual SO101 robot arms."""
    typer.echo("\n=== Policy Inference ===\n")

    # Load global config for hardware settings
    config = load_config()
    validate_config(config)

    # Load local config for inference settings
    config_path = INFERENCE_DIR / config_file
    if not config_path.exists():
        typer.echo(f"Config file not found: {config_path}")
        raise typer.Exit(1)
    local_config = load_config(config_path)
    config.update(local_config)

    # Get configurations
    inference_cfg = get_inference_config(config)
    cameras_cfg = get_camera_config(config)

    # Override display from config if not specified
    if "display" in inference_cfg:
        display = inference_cfg["display"] and display

    # Merge CLI record flag with config
    record = record or inference_cfg["record"]
    inference_cfg["record"] = record

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
        n_action_steps=inference_cfg.get("n_action_steps"),
        rename_map=inference_cfg.get("rename_map"),
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

    # Set up recording dataset if enabled
    rec_dataset = None
    if record:
        # Derive record repo from policy path for traceability
        policy_name = Path(inference_cfg["policy_repo_id"]).name
        rec_repo_id = inference_cfg["record_repo_id"] or f"{policy_name}-inference"
        rec_local_path = get_local_dataset_path(rec_repo_id)

        # Check if existing dataset has recorded episodes (can resume)
        # or is a stale empty dir from a previous crash (must recreate)
        info_path = rec_local_path / "meta" / "info.json"
        has_episodes = False
        if info_path.exists():
            info = json.loads(info_path.read_text())
            has_episodes = info.get("total_episodes", 0) > 0

        if has_episodes:
            typer.echo(f"\nResuming recording dataset: {rec_repo_id}")
            rec_dataset = LeRobotDataset(
                repo_id=rec_repo_id,
                root=rec_local_path,
                streaming_encoding=True,
            )
            rec_dataset.start_image_writer(num_processes=0, num_threads=4 * len(cameras_cfg))
            rec_dataset.episode_buffer = rec_dataset.create_episode_buffer()
        else:
            # Remove stale empty dataset dir from a previous crash
            if rec_local_path.exists():
                shutil.rmtree(rec_local_path)
            # Extract non-default features from the training dataset
            recording_features = {
                k: v for k, v in dataset.features.items() if k not in DEFAULT_FEATURES
            }

            typer.echo(f"\nCreating recording dataset: {rec_repo_id}")
            rec_dataset = LeRobotDataset.create(
                repo_id=rec_repo_id,
                fps=fps,
                root=rec_local_path,
                robot_type=robot.name,
                features=recording_features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=4 * len(cameras_cfg),
                streaming_encoding=True,
            )

        # Embed metadata for reproducibility
        git_info = get_git_info()
        if git_info["git_hash"]:
            rec_dataset.meta.info["git_hash"] = git_info["git_hash"]
            rec_dataset.meta.info["git_branch"] = git_info["git_branch"]
            rec_dataset.meta.info["git_dirty"] = git_info["git_dirty"]
        rec_dataset.meta.info["source_policy"] = inference_cfg["policy_repo_id"]
        rec_dataset.meta.info["source_dataset"] = inference_cfg["dataset_repo_id"]
        typer.echo("Recording dataset ready.")

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

        # Timers for control rate monitoring
        t_loop = TimerManager("loop", log=False)
        t_obs = TimerManager("obs", log=False)
        t_policy = TimerManager("policy", log=False)
        t_action = TimerManager("action", log=False)

        dt = 1.0 / fps
        next_tick = time.perf_counter() + dt
        last_log_time = time.perf_counter()

        encoding_ctx = (
            VideoEncodingManager(rec_dataset) if rec_dataset else contextlib.nullcontext()
        )
        with encoding_ctx:
            while not events["stop_recording"]:
                t_loop.start()

                # 1. Get observation from robot
                with t_obs:
                    obs = robot.get_observation()
                    obs_processed = robot_observation_processor(obs)

                # 2. Build observation frame for policy
                observation_frame = build_dataset_frame(
                    dataset.features,
                    obs_processed,
                    prefix=OBS_STR,
                )

                # 3. Run policy inference
                with t_policy:
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
                with t_action:
                    robot.send_action(robot_action_processor((robot_action, obs)))

                # 6. Record frame if recording
                if rec_dataset is not None:
                    action_frame = build_dataset_frame(
                        dataset.features, robot_action, prefix=ACTION
                    )
                    frame = {**observation_frame, **action_frame, "task": task}
                    rec_dataset.add_frame(frame)

                # 7. Log visualization if enabled
                if display:
                    log_observation_and_action(
                        visualizer=visualizer,
                        observation=obs,
                        action=robot_action,
                        use_degrees=True,
                    )

                step += 1

                # Sync to exact target FPS using wall-clock schedule
                t_loop.stop()
                now = time.perf_counter()
                precise_sleep(max(next_tick - now, 0.0))
                next_tick += dt

                # Log control rate every second
                if step % fps == 0:
                    log_now = time.perf_counter()
                    actual_hz = fps / (log_now - last_log_time)
                    last_log_time = log_now
                    elapsed = time.time() - start_time
                    typer.echo(
                        f"  {elapsed:5.1f}s | {step} steps | "
                        f"{actual_hz:.1f}/{fps} Hz | "
                        f"obs {t_obs.avg * 1e3:.1f}ms  "
                        f"policy {t_policy.avg * 1e3:.1f}ms  "
                        f"action {t_action.avg * 1e3:.1f}ms  "
                        f"p95 {t_loop.percentile(95) * 1e3:.1f}ms"
                    )
                    if actual_hz < fps * 0.95:
                        typer.echo(
                            f"  WARNING: control rate {actual_hz:.1f} Hz below target {fps} Hz"
                        )
                    t_loop.reset()
                    t_obs.reset()
                    t_policy.reset()
                    t_action.reset()

        # Save episode after clean exit
        if rec_dataset is not None and step > 0:
            typer.echo("Saving recorded episode...")
            rec_dataset.save_episode()

        log_say("Inference stopped.", play_sounds=True)

    except KeyboardInterrupt:
        typer.echo("\n\nInference interrupted by user.")
        # Save partial episode on interrupt
        if rec_dataset is not None and step > 0:
            typer.echo("Saving partial recorded episode...")
            try:
                rec_dataset.save_episode()
            except Exception as e:
                typer.echo(f"Warning: failed to save partial episode: {e}")

    finally:
        if rec_dataset is not None:
            rec_dataset.finalize()

        if push_to_hub and rec_dataset is not None:
            typer.echo("\nPushing recorded dataset to Hugging Face Hub...")
            rec_dataset.push_to_hub()
            typer.echo("Dataset uploaded successfully!")

        if listener:
            listener.stop()

        stow_and_disconnect(robot)

        # Save Rerun recording
        if display and (rrd_path := save_rrd()):
            typer.echo(f"Rerun recording saved to: {rrd_path}")

    total_time = time.time() - start_time
    avg_hz = step / total_time if total_time > 0 else 0
    typer.echo(
        f"\nInference complete. {total_time:.1f}s, {step} steps, "
        f"avg {avg_hz:.1f} Hz (target {fps} Hz)"
    )


if __name__ == "__main__":
    app()
